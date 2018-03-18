# -*- coding: utf-8 -*-

#
# Imports
#
try:
	from .crc_native import crc32c_update
except ImportError:
	raise ImportError(
"""\
Could not import native Python module crc_native.

Are you currently importing from within the source directory rather than the
install directory? The compiled native module is not built within the source
directory tree.\
""")




#
# CRC32C Castagnoli Implementation
#
class CRC32C(object):
	def __init__(self, crc=None):
		self.init(crc)
	def init(self, crc=None):
		self.crc = int(~0 if crc is None else crc) & 0xFFFFFFFF
		return self
	def update(self, data):
		self.crc = crc32c_update(self.crc, bytes(data))
		return self
	def finalize(self):
		return ~self.crc & 0xFFFFFFFF

