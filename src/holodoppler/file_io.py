"""
File I/O for .holo and .cine files
"""

import os
import json
import numpy as np
import traceback

try:
    import cinereader
    CINE_AVAILABLE = True
except ImportError:
    cinereader = None
    CINE_AVAILABLE = False


class HoloFileReader:
    """Reader for .holo files"""
    
    HEADER_SIZE = 64
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.fid = None
        self.file_header = None
        self.file_footer = None
        
    def open(self):
        if self.fid is not None:
            self.close()
        self.fid = open(self.file_path, "rb")
        self._read_header()
        self._read_footer()
        
    def close(self):
        if self.fid is not None:
            self.fid.close()
            self.fid = None
    
    def _read_header(self):
        header = self.fid.read(self.HEADER_SIZE)
        self.file_header = {
            "magic_number": ''.join(list(map(chr, header[0:4]))),
            "version": int.from_bytes(header[4:6], "little"),
            "bit_depth": int.from_bytes(header[6:8], "little"),
            "width": int.from_bytes(header[8:12], "little"),
            "height": int.from_bytes(header[12:16], "little"),
            "num_frames": int.from_bytes(header[16:20], "little"),
            "total_size": int.from_bytes(header[20:28], "little"),
            "endianness": header[28]
        }
    
    def _read_footer(self):
        w, h = self.file_header["width"], self.file_header["height"]
        num_frames = self.file_header["num_frames"]
        offset = w * h * num_frames + self.HEADER_SIZE
        self.fid.seek(offset)
        footer_bytes = self.fid.read()
        if footer_bytes:
            try:
                self.file_footer = json.loads(footer_bytes.decode("utf-8"))
            except Exception:
                self.file_footer = {}
        else:
            self.file_footer = {}
    
    def read_frames(self, first_frame, frame_size):
        """Read frames from .holo file (returns numpy array)"""
        try:
            byte_begin = (
                self.HEADER_SIZE
                + self.file_header["width"] * self.file_header["height"] 
                * first_frame * self.file_header["bit_depth"] // 8
            )
            byte_size = (
                self.file_header["width"] * self.file_header["height"] 
                * frame_size * self.file_header["bit_depth"] // 8
            )
            
            self.fid.seek(byte_begin)
            raw_bytes = self.fid.read(byte_size)
            
            if self.file_header["bit_depth"] == 8:
                utyp = np.uint8
            elif self.file_header["bit_depth"] == 16:
                utyp = np.uint16
            else:
                raise RuntimeError("Unsupported bit depth")
            
            if self.file_header["endianness"] == 1:
                utyp = utyp.newbyteorder('<')
            
            out = np.frombuffer(raw_bytes, dtype=utyp)
            out = out.reshape(
                (frame_size, self.file_header["height"], self.file_header["width"]),
                order="C"
            )
            return out.astype(np.float32)
        except Exception:
            traceback.print_exc()
            return None


class CineFileReader:
    """Reader for .cine files"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = None
        self.metadata_json = None
        
        if not CINE_AVAILABLE:
            raise RuntimeError("cinereader module not available")
    
    def open(self):
        self.metadata = cinereader.read_metadata(self.file_path)
        self.metadata_json = dict(self.metadata.__dict__)
    
    def close(self):
        pass  # cinereader doesn't need explicit close
    
    def read_frames(self, first_frame, frame_size):
        _, images, _ = cinereader.read(
            self.file_path, 
            self.metadata.FirstImageNo + first_frame, 
            frame_size
        )
        return np.stack(images, axis=0).astype(np.float32)


class FileReaderFactory:
    """Factory to create appropriate file reader"""
    
    @staticmethod
    def create(file_path):
        _, ext = os.path.splitext(file_path)
        
        if ext == ".holo":
            reader = HoloFileReader(file_path)
        elif ext == ".cine":
            reader = CineFileReader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        reader.ext = ext
        return reader