import os
import json

import numpy as np
import traceback
import cinereader

class FileReader():
    # ------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------

    def load_file(self, file_path):

        if self.fid is not None:
            self.fid.close()

        _, ext = os.path.splitext(file_path)

        self.file_path = file_path

        if ext == ".holo":
            self.ext = ext
            def _extract_holo_footer(f, w, h, numframes):
                offset = w * h * numframes + 64
                f.seek(offset)
                footer_bytes = f.read()
                if not footer_bytes:
                    return {}
                try:
                    footer_str = footer_bytes.decode("utf-8")
                    return json.loads(footer_str)
                except Exception:
                    return {}
            self.fid = open(self.file_path, "rb")
            header = self.fid.read(self.HOLO_HEADER_SIZE)
            file_header = dict()
            file_header["magic_number"] = ''.join(list(map(chr, header[0:4])))
            file_header["version"] = int.from_bytes(header[4:6], "little")
            file_header["bit_depth"] = int.from_bytes(header[6:8], "little")
            file_header["width"] = int.from_bytes(header[8:12], "little")
            file_header["height"] = int.from_bytes(header[12:16], "little")
            file_header["num_frames"] = int.from_bytes(header[16:20], "little")
            file_header["total_size"] = int.from_bytes(header[20:28], "little")
            file_header["endianness"] = header[28]

            self.file_footer = _extract_holo_footer(self.fid, file_header["width"], file_header["height"], file_header["num_frames"])
            self.file_header = file_header
            self.read_frames = self.read_frames_holo

        elif ext == ".cine":
            self.ext = ext
            self.cine_metadata = cinereader.read_metadata(file_path)
            self.cine_metadata_json = dict(self.cine_metadata.__dict__)
            self.read_frames = self.read_frames_cine

    def _close_file(self):
        if self.fid is not None:
            self.fid.close()

    # ------------------------------------------------------------
    # Reading frames (CPU only, then transferred if needed)
    # ------------------------------------------------------------

    def read_frames_cine(self, first_frame, frame_size):
        _, images, _ = cinereader.read(self.file_path, self.cine_metadata.FirstImageNo + first_frame, frame_size)
        frames = np.stack(images, axis=0)
        return self._to_backend(frames)
    
    def read_frames_holo(self, first_frame, frame_size, fid = None):
        
        try:

            if fid is None:
                fid = self.fid
            
            byte_begin = (
                self.HOLO_HEADER_SIZE
                + self.file_header["width"] * self.file_header["height"] * first_frame * self.file_header["bit_depth"] // 8
            )

            byte_size = self.file_header["width"] * self.file_header["height"] * frame_size * self.file_header["bit_depth"] // 8

            fid.seek(byte_begin)
            raw_bytes = fid.read(byte_size)

            if self.file_header["bit_depth"] == 8:
                utyp = np.uint8
            elif self.file_header["bit_depth"] == 16:
                utyp = np.uint16
            else:
                raise RuntimeError("Unsupported bit depth : Supported bit depth are 8 bits or 16 bits")

            if self.file_header["endianness"] == 1:
                utyp = utyp.newbyteorder('<')

            out = np.frombuffer(raw_bytes, dtype=utyp)

            out = out.reshape(
                (frame_size,self.file_header["height"], self.file_header["width"]),
                order="C"
            )
            
            return self._to_backend(out).astype(self.xp.float32)

        except Exception:
            traceback.print_exc()
            return None