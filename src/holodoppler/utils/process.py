import matplotlib
import numpy as np
import traceback
import h5py
from tqdm import tqdm
import cv2
import json
import os
from cupy.cuda.nvtx import RangePush, RangePop
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as np_gaussian_filter
from scipy.ndimage import gaussian_filter1d
import cupy as cp

class Processor():
    # ------------------------------------------------------------
    # Full video processing
    # ------------------------------------------------------------

    def process_moments_(self, parameters, h5_path = None, mp4_path = None, return_numpy = False, holodoppler_path = False):
        
        batch_size = parameters["batch_size"]
        batch_stride = parameters["batch_stride"]
        first_frame = parameters["first_frame"]
        end_frame = parameters["end_frame"]
        if end_frame <= 0:
            if self.ext == ".holo":
                end_frame = self.file_header["num_frames"]
            elif self.ext == ".cine":
                end_frame = self.cine_metadata.ImageCount
        
        # please do not remove it is good
        if batch_stride >= (end_frame-first_frame):
            if batch_size <= (end_frame-first_frame):
                num_batch = 1
            else:
                num_batch = 0
        else:
            num_batch = int((end_frame-first_frame) / batch_stride)

        out_list = []

        if num_batch <= 0:
            return None
        
        if parameters["debug"]:
            import threading
            import queue
            
            self.init_plot_debug(parameters)

            # --- create plotting worker ---

            debug_results = {}
            res_store = {}   # shared storage
            lock = threading.Lock()

            def plotting_worker(in_q, stop_event):
                while not stop_event.is_set() or not in_q.empty():
                    try:
                        i = in_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    with lock:
                        res = res_store.pop(i)  # remove to free memory early

                    out = self.plot_debug(res, i)

                    with lock:
                        debug_results[i] = out

                    in_q.task_done()

            debugin_queue = queue.Queue(maxsize=14)
            stop_event = threading.Event()

            debug_thread = threading.Thread(
                target=plotting_worker,
                args=(debugin_queue, stop_event),
                daemon=True
            )
            debug_thread.start()
        
        if parameters["image_registration"]:
            frames = self.read_frames(first_frame, parameters["batch_size_registration"]) # the first frame to be rendered
            M0_reg = self.render_moments(parameters, frames = frames)["M0"] # render the first frame to be used as reference for the registration
            M0_reg = self._flatfield(M0_reg, parameters["registration_flatfield_gw"])
            reg_list = [None] * num_batch
        else:
            M0_reg = None
            
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and parameters["shack_hartmann_zernike_fit"]:
            coefs_list = [None] * num_batch

        if self.backend == "cupy":
            stream_h2d = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)

            # --- Prefetch first batch ---
            frames_next = self.read_frames(
                first_frame,
                parameters["batch_size"]
            )

            with stream_h2d:
                d_frames_next = cp.asarray(frames_next) 

            for i in tqdm(range(num_batch)):

                # Swap buffers
                d_frames = d_frames_next

                # --- Prefetch next batch (CPU side) ---
                if i + 1 < num_batch:
                    
                    with stream_h2d:
                        d_frames_next = self.read_frames(
                        first_frame + (i + 1) * parameters["batch_stride"],
                        parameters["batch_size"]
                    )

                # --- Compute current batch ---
                with stream_compute:
                    res = self.render_moments(parameters, frames=d_frames, registration_ref=M0_reg)

                if res is None:
                    break

                M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                
                if parameters["debug"]:
                    # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                    with lock:
                        res_store[i] = res
                    debugin_queue.put((i))
                
                stream_compute.synchronize()
                
                out_list.append(
                    cp.stack([M0, M1, M2], axis=2)
                )
                if "coefs" in res:
                    coefs_list[i] = res["coefs"]
                if "registration" in res:
                    reg_list[i] = res["registration"]
                

            # Ensure transfers complete
            stream_h2d.synchronize()

            cp.cuda.Device().synchronize()

        elif self.backend == "cupyRAM":
            import queue
            import threading
            import time
            from collections import deque
            
            # Profiling data collection
            profile_data = {
                'read_times': deque(maxlen=100),  # Keep last 100 reads
                'transfer_times': deque(maxlen=100),
                'compute_times': deque(maxlen=100),
                'registration_times': deque(maxlen=100),
                'queue_wait_times': deque(maxlen=100),
                'total_batch_times': deque(maxlen=100),
                'queue_sizes': deque(maxlen=100),
            }
            
            # Create queues for frames and results
            frame_queue = queue.Queue(maxsize=4)  # Configurable queue depth
            
            # Control flags
            stop_reader = threading.Event()
            reader_done = threading.Event()
            
            # Reader thread function - continuously reads frames
            def reader_thread_func():
                frame_idx = first_frame
                batch_num = 0
                
                while batch_num < num_batch and not stop_reader.is_set():
                    read_start = time.perf_counter()
                    
                    # Read frames
                    frames = self.read_frames(frame_idx, batch_size)
                    
                    read_time = time.perf_counter() - read_start
                    profile_data['read_times'].append(read_time)
                    
                    # Put into queue (will block if queue is full)
                    queue_start = time.perf_counter()
                    frame_queue.put((batch_num, frame_idx, frames))
                    queue_time = time.perf_counter() - queue_start
                    profile_data['queue_wait_times'].append(queue_time)
                    
                    # Track queue size
                    profile_data['queue_sizes'].append(frame_queue.qsize())
                    
                    batch_num += 1
                    frame_idx += batch_stride
                
                reader_done.set()
            
            # Start reader thread
            reader_thread = threading.Thread(target=reader_thread_func, daemon=True)
            reader_thread.start()
            
            # CUDA streams for async operations
            stream_h2d = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)
            stream_registration = cp.cuda.Stream(non_blocking=True) if parameters["image_registration"] else None
            
            # Pre-start with first batch
            first_batch_num, first_frame_idx, first_frames = frame_queue.get()
            
            # Async transfer to GPU
            transfer_start = time.perf_counter()
            with stream_h2d:
                d_frames_current = cp.asarray(first_frames)
            stream_h2d.synchronize()
            transfer_time = time.perf_counter() - transfer_start
            profile_data['transfer_times'].append(transfer_time)
            
            current_batch_num = first_batch_num
            current_frame_idx = first_frame_idx
            d_frames = d_frames_current
            
            # Main processing loop
            processed_batches = 0
            next_batch_prefetched = False
            
            for i in tqdm(range(num_batch)):
                batch_start_time = time.perf_counter()
                
                # Check if we need to get next batch from queue
                if not next_batch_prefetched and i > 0:
                    queue_wait_start = time.perf_counter()
                    next_batch_num, next_frame_idx, next_frames = frame_queue.get()
                    queue_wait_time = time.perf_counter() - queue_wait_start
                    profile_data['queue_wait_times'].append(queue_wait_time)
                    
                    # Async transfer next batch
                    transfer_start = time.perf_counter()
                    with stream_h2d:
                        d_frames_next = cp.asarray(next_frames)
                    stream_h2d.synchronize()
                    transfer_time = time.perf_counter() - transfer_start
                    profile_data['transfer_times'].append(transfer_time)
                    
                    next_batch_prefetched = True
                
                # Compute current batch
                compute_start = time.perf_counter()
                with stream_compute:
                    res = self.render_moments(parameters, frames=d_frames, registration_ref=M0_reg)
                
                if res is None:
                    break
                
                M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                if parameters["debug"]:
                    # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                    with lock:
                        res_store[i] = res
                    debugin_queue.put((i))
                if "coefs" in res:
                    coefs_list[i] = res["coefs"]
                if "registration" in res:
                    reg_list[i] = res["registration"]
                    
                compute_time = time.perf_counter() - compute_start
                profile_data['compute_times'].append(compute_time)
                
                # Ensure compute is done
                stream_compute.synchronize()
                
                # Store result
                out_list.append(cp.stack([M0, M1, M2], axis=2))
                
                # Swap buffers for next iteration
                if next_batch_prefetched and i + 1 < num_batch:
                    d_frames = d_frames_next
                    current_batch_num = next_batch_num
                    next_batch_prefetched = False
                    processed_batches += 1
                
                # Profile total batch time
                batch_time = time.perf_counter() - batch_start_time
                profile_data['total_batch_times'].append(batch_time)
            
            # Signal reader to stop and wait for completion
            stop_reader.set()
            reader_thread.join(timeout=5)
            
            # Ensure all CUDA operations complete
            stream_h2d.synchronize()
            stream_compute.synchronize()
            if stream_registration:
                stream_registration.synchronize()
            cp.cuda.Device().synchronize()
            
            # Print profiling summary
            if parameters.get("enable_profiling", True):
                print("\n" + "="*60)
                print("PROFILING SUMMARY - cupyRAM Backend")
                print("="*60)

                import numpy as np
                
                def print_stats(name, times):
                    if len(times) > 0:
                        avg = np.mean(times) * 1000  # Convert to ms
                        std = np.std(times) * 1000
                        min_t = np.min(times) * 1000
                        max_t = np.max(times) * 1000
                        print(f"{name:20s}: Avg={avg:6.2f}ms ±{std:5.2f}ms, Min={min_t:6.2f}ms, Max={max_t:6.2f}ms")
                    else:
                        print(f"{name:20s}: No data")
                
                print_stats("File Read Time", profile_data['read_times'])
                print_stats("Queue Wait Time", profile_data['queue_wait_times'])
                print_stats("H2D Transfer Time", profile_data['transfer_times'])
                print_stats("Compute Time", profile_data['compute_times'])
                print_stats("Registration Time", profile_data['registration_times'])
                print_stats("Total Batch Time", profile_data['total_batch_times'])
                
                if len(profile_data['queue_sizes']) > 0:
                    avg_queue = np.mean(profile_data['queue_sizes'])
                    max_queue = np.max(profile_data['queue_sizes'])
                    print(f"\nQueue Statistics:")
                    print(f"  Average Queue Size: {avg_queue:.1f}")
                    print(f"  Max Queue Size: {max_queue}")
                
                # Calculate throughput
                if len(profile_data['total_batch_times']) > 0:
                    total_time = np.sum(profile_data['total_batch_times'])
                    total_batches = len(profile_data['total_batch_times'])
                    total_frames = total_batches * batch_size
                    print(f"\nThroughput:")
                    print(f"  Batches/sec: {total_batches/total_time:.2f}")
                    print(f"  Frames/sec: {total_frames/total_time:.2f}")
                    
                    # Utilization metrics
                    compute_total = np.sum(profile_data['compute_times'])
                    read_total = np.sum(profile_data['read_times'])
                    transfer_total = np.sum(profile_data['transfer_times'])
                    
                    if total_time > 0:
                        print(f"\nUtilization (of total time):")
                        print(f"  Compute: {compute_total/total_time*100:.1f}%")
                        print(f"  File I/O: {read_total/total_time*100:.1f}%")
                        print(f"  H2D Transfer: {transfer_total/total_time*100:.1f}%")
                        
                        # Overlap efficiency
                        ideal_time = max(compute_total, read_total + transfer_total)
                        overlap_efficiency = ideal_time / total_time * 100 if total_time > 0 else 0
                        print(f"  Overlap Efficiency: {overlap_efficiency:.1f}%")
                
                print("="*60 + "\n")

        elif self.backend == "numpy multiprocessing":

            raise NotImplementedError("Multiprocessing backend is not implemented yet. Please use 'cupy' or 'cupyRAM' for GPU acceleration or 'numpy' for CPU serial execution.")

        else:
            for i in tqdm(range(num_batch)):

                try:

                    frames = self.read_frames(first_frame + i * parameters["batch_stride"] , parameters["batch_size"])

                    res = self.render_moments(parameters, frames = frames, registration_ref=M0_reg)

                    if res is None:
                        break

                    M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                    if parameters["debug"]:
                        # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                        with lock:
                            res_store[i] = res
                        debugin_queue.put((i))
                    if "coefs" in res:
                        coefs_list[i] = res["coefs"]
                    if "registration" in res:
                        reg_list[i] = res["registration"]

                    out_list.append(
                        self.xp.stack([M0, M1, M2], axis=2)
                    )

                except Exception:
                    traceback.print_exc()
                    break

            if len(out_list) == 0:
                return None
        
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and all(coefs is not None for coefs in coefs_list):
            zernike_coefs = self._to_numpy(self.xp.array(coefs_list))
        else:
            zernike_coefs = None
        
        vid = self.xp.stack(out_list, axis=3) 
        
        # ------------------------------------------------------------
        # Debug handling
        # ------------------------------------------------------------
        import time
        t0 = time.perf_counter()
        if parameters["debug"]:
            # --- build streams automatically ---
            streams = {k: [None]*len(debug_results) for k in self.debug_plotters.keys()}

            for i, dic in debug_results.items():
                for key, img in dic.items():
                    if parameters["square"] and key in ["montage", "M0notfixed"]:
                        m = max(img.shape[0], img.shape[1])
                        img = self._resize(img, m, m)

                    streams[key][i] = img

            # stack only non-empty streams
            import numpy as np
            vid_debug = [
                np.stack([img for img in stream if img is not None], axis=2)
                for stream in streams.values()
                if any(img is not None for img in stream)
            ]
            debugin_queue.join()
            stop_event.set()
            debug_thread.join()

        else:
            vid_debug = None
        t_debug = time.perf_counter() - t0
        # print(f"[TIMING] debug handling: {t_debug:.3f}s")

        # ------------------------------------------------------------
        # Zernike coefficients
        # ------------------------------------------------------------
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and all(coefs is not None for coefs in coefs_list):
            zernike_coefs = self._to_numpy(self.xp.array(coefs_list))
        else:
            zernike_coefs = None

        # ------------------------------------------------------------
        # Temporal accumulation
        # ------------------------------------------------------------
        # t0 = time.perf_counter()
        # if parameters["accumulation"] > 1:
        #     acc = parameters["accumulation"] 
        #     ny, nx, nimgs, nt = vid.shape
        #     vid = self.xp.reshape(vid[:,:,:,:(nt//acc)*acc], (ny, nx, nimgs, nt//acc, acc)) @ self.xp.ones(acc)
        # t_acc = time.perf_counter() - t0
        # print(f"[TIMING] accumulation: {t_acc:.3f}s")

        # ------------------------------------------------------------
        # Spatial transformations (square, transpose, flips)
        # ------------------------------------------------------------
        import numpy as np
        t0 = time.perf_counter()
        if parameters["square"]:
            m = max(vid.shape[0], vid.shape[1])
            vid = self._to_numpy(vid).astype(np.float64)
            vid = self._resize(vid, m, m)
        if parameters["transpose"]:
            vid = self.xp.transpose(vid, axes=(1, 0, 2, 3))
        if parameters["flip_x"]:
            vid = self.xp.flip(vid, axis=1)
        if parameters["flip_y"]:
            vid = self.xp.flip(vid, axis=0)
        t_spatial = time.perf_counter() - t0
        # print(f"[TIMING] spatial transforms: {t_spatial:.3f}s")

        # ------------------------------------------------------------
        # Convert to numpy if needed
        # ------------------------------------------------------------
        t0 = time.perf_counter()
        if (h5_path is not None) or (mp4_path is not None) or holodoppler_path:
            vid = self._to_numpy(vid)
        t_to_numpy = time.perf_counter() - t0
        # print(f"[TIMING] to_numpy conversion: {t_to_numpy:.3f}s")
        
        # -------------------------------------------------------------
        # Close file if needed
        # -------------------------------------------------------------
        self._close_file()

        # ------------------------------------------------------------
        # Saving outputs (h5, mp4, holodoppler)
        # ------------------------------------------------------------
        
        def save_to_h5path(h5_path, v, parameters, reg_list = None, zernike_coefs = None, git_commit = None):
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("moment0", data=v[:, :, :, 0])
                f.create_dataset("moment1", data=v[:, :, :, 1])
                f.create_dataset("moment2", data=v[:, :, :, 2])
                f.create_dataset("HD_parameters", data=json.dumps(parameters))
                if parameters["image_registration"]:
                    f.create_dataset("registration", data=self._to_numpy(self.xp.array(reg_list)))
                    
                if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and zernike_coefs is not None:
                    f.create_dataset("zernike_coefs_radians", data=self._to_numpy((zernike_coefs).astype(np.float64)))

                def json_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Type {type(obj)} not serializable")

                if self.ext == ".holo":
                    f.create_dataset("holo_header", data=json.dumps(self.file_header, default=json_serializer))
                    f.create_dataset("holo_footer", data=json.dumps(self.file_footer, default=json_serializer))
                elif self.ext == ".cine":
                    f.create_dataset("cine_metadata", data=json.dumps(self.cine_metadata_json, default=json_serializer))
                if git_commit is not None:
                    f.create_dataset("git_commit", data=git_commit)
                    
                    
        t_save = 0.0

        if h5_path is not None:
            tt = time.perf_counter()
            try:
                import subprocess
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            except Exception:
                git_commit = None
            save_to_h5path(h5_path, v, parameters, reg_list = None, zernike_coefs = None, git_commit = git_commit)
            t_save += time.perf_counter() - tt
            # print(f"[TIMING] save HDF5: {time.perf_counter()-tt:.3f}s")

        if mp4_path is not None:
            tt = time.perf_counter()
            
            # save m0 as mp4
            def normalize(arr):
                arr = arr.astype(np.float32)
                lo, hi = arr.min(), arr.max()
                return ((arr - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else arr.astype(np.uint8)
            def write_video(path, frames, fps):
                h, w, n = frames.shape[0], frames.shape[1], frames.shape[2]
                out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
                for i in range(n):
                    out.write(normalize(frames[:, :, i]))
                out.release()
            fps = num_batch / (end_frame - first_frame) * parameters["sampling_freq"]
            write_video(mp4_path, vid[:, :, 0, :], min(fps, 65), "mp4v")
            t_save += time.perf_counter() - tt
            # print(f"[TIMING] save MP4: {time.perf_counter()-tt:.3f}s")

        if holodoppler_path:
            tt = time.perf_counter()
            
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            dir_name = f"{base_name}"
            parent_dir = os.path.dirname(self.file_path)
            holodoppler_dir_name = f"{dir_name}_HD"
            holodoppler_path = os.path.join(parent_dir,dir_name, holodoppler_dir_name)
            os.makedirs(holodoppler_path, exist_ok=True)
            # make a png, mp4 json and h5 sub directories with their respective content
            png_dir = os.path.join((holodoppler_path), "png")
            mp4_dir = os.path.join((holodoppler_path), "mp4")
            avi_dir = os.path.join((holodoppler_path), "avi")
            json_dir = os.path.join((holodoppler_path), "json")
            h5_dir = os.path.join((holodoppler_path), "h5")
            print(f"Saving output to: {holodoppler_path}")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(mp4_dir, exist_ok=True)
            os.makedirs(avi_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(h5_dir, exist_ok=True)
            
            # save m0 as mp4 and avi
            def normalize(arr):
                arr = arr.astype(np.float32)
                lo, hi = arr.min(), arr.max()
                return ((arr - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else arr.astype(np.uint8)

            def temporal_gaussian(arr, sigma):
                if sigma == 0 :
                    return arr
                return gaussian_filter1d(arr.astype(np.float32), sigma=sigma, axis=2)

            def write_video(path, frames, fps, fourcc, is_color=False):
                h, w, n = frames.shape[0], frames.shape[1], frames.shape[2]
                out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h), isColor=is_color)
                for i in range(n):
                    out.write(frames[:, :, i] if frames.ndim == 3 else frames[:, :, i, :])
                out.release()

            def save_pair(stem, frames, fps, mp4_dir, avi_dir, sigma = 4.0, is_color=False, save_png=True):
                frames = temporal_gaussian(frames, sigma) # removing for clarity only raw output
                frames = normalize(frames)
                
                # print(frames.shape, frames.dtype, type(frames))
                write_video(os.path.join(mp4_dir, f"{stem}.mp4"), frames, min(fps, 65), "mp4v", is_color)
                write_video(os.path.join(avi_dir, f"{stem}.avi"), frames, min(fps, 65), "MJPG", is_color)
                if save_png:
                    plt.imsave(os.path.join(png_dir, f"{stem}.png"), np.mean(frames, axis=2), cmap="gray")

            fps = num_batch / (end_frame - first_frame) * parameters["sampling_freq"]

            save_pair("moment_0", vid[:, :, 0, :], fps, mp4_dir, avi_dir, sigma=0)
            save_pair("moment_1", vid[:, :, 1, :], fps, mp4_dir, avi_dir, sigma=0)
            save_pair("moment_2", vid[:, :, 2, :], fps, mp4_dir, avi_dir, sigma=0)

            if parameters["debug"]:
                def flatfield3D(arr, gw):
                    if arr.ndim != 3:
                        raise ValueError("Input array must be 3D")
                    if gw <= 1:
                        return arr
                    blurred = np_gaussian_filter(arr, sigma=(gw, gw, 1))
                    blurred[blurred == 0] = 1
                    return arr / blurred
                save_pair("moment_0_slidingavg_flatfield", flatfield3D(vid[:, :, 0, :], parameters["registration_flatfield_gw"]), fps, mp4_dir, avi_dir, sigma=1.50)
            

            if parameters["debug"] and vid_debug is not None:
                for idx, v in enumerate(vid_debug):
                    save_pair(f"debug_{list(self.debug_plotters.keys())[idx]}", v, fps, mp4_dir, avi_dir, sigma=0, is_color=v.ndim == 4, save_png=False)

            # save json
            with open(os.path.join(json_dir, "parameters.json"), "w") as f:
                json.dump(parameters, f, indent=4)
                
            # get git info 
            # if git is available write the current commit hash
            try:
                import subprocess
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                git_txt = "Git commit hash: " + git_commit + "\n"
                # check if there are uncommited changes and add a warning if there are
                git_status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
                if git_status:
                    git_txt += "Warning: There are uncommited changes in the repository, the results may not be reproducible\n"
            except Exception:
                git_txt = "Git commit hash: Not available\n"
            # save h5
            save_to_h5path(os.path.join(h5_dir, f"{holodoppler_dir_name}_output.h5"), np.permute_dims(vid, (3, 1, 0, 2)), parameters, reg_list if parameters["image_registration"] else None, zernike_coefs, git_commit=git_txt)
            # add a version.txt file with the version of the holodoppler pipeline used
            with open(os.path.join(holodoppler_path, "git_version.txt"), "w") as f:
                f.write(f"Python:\n")
                f.write(f"Holodoppler pipeline version: {self.pipeline_version}\n")
                f.write(f"Holodoppler backend: {self.backend}\n") 
                f.write(f"{git_txt}")
                
            with open(os.path.join(holodoppler_path, "version_holodoppler.txt"), "w") as f:
                f.write(f"py 0.1.0")
            if self.ext == ".holo":
                with open(os.path.join(holodoppler_path, "version_holovibes.txt"), "w") as f:
                    f.write(f"{self.file_footer.get('info',{}).get('holovibes_version', 'unknown')}")
                    
            t_save += time.perf_counter() - tt
            # print(f"[TIMING] holodoppler outputs: {time.perf_counter()-tt:.3f}s")
            
        # Dispose of GPU arrays to free memory
        if self.backend in ["cupy", "cupyRAM"]:
            del d_frames
            if 'd_frames_next' in locals():
                del d_frames_next
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
        plt.close('all') # close any open figures to free memory

        print(f"total saving time: {t_save:.3f}s")

        if return_numpy:
            return self._to_numpy(vid)