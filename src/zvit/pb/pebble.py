# -*- coding: utf-8 -*-

#
# Imports
#
import struct

from   .crc   import CRC32C



#
# Basic ProtoBuf emitter utilities
#
class TFCRC32C(CRC32C):
	"""
	Custom TF CRC32C checksum variant.
	
	The only difference is in the finalization function.
	"""
	
	def finalize(self):
		"""
		TF does a special "masked" finalization on CRC32C's output. This
		involves a rotate-right by 15 bits and the addition of 0xa282ead8.
		
		What this accomplishes is unknown, aside from decreasing compatibility.
		"""
		
		v = super(TFCRC32C, self).finalize()
		return ((v>>15 | v<<17) + 0xa282ead8) & 0xFFFFFFFF

def tfcrc32c(buf):
	return TFCRC32C().update(buf).finalize()


#
# The ProtoBuf message class
#
class PebbleMessage(object):
	"""Base class for a ProtoBuf message."""
	
	__protobuf__ = []
	
	def __init__(self, **kwargs):
		for name, value in kwargs.items():
			setattr(self, name, value)
	
	def __setattr__(self, name, value):
		#
		# Special: Assigning None to a PebbleElement-managed attribute deletes it.
		# Special: Handle erasure of alternatives in oneof elements. Assigning
		#          to one alternative erases from all others.
		#
		for elem in self.iterelems():
			if isinstance(elem, tuple):
				for oneofElem in elem:
					if name == oneofElem.tag:
						if value is None:
							delattr(self, oneofElem.tag)
						else:
							for oneofElemDel in elem:
								delattr(self, oneofElemDel.tag)
							super().__setattr__(name, value)
			else:
				if name == elem.tag:
					if value is None:
						delattr(self, elem.tag)
					else:
						super().__setattr__(name, value)
		
		super().__setattr__(name, value)
	
	def asByteArray(self):
		b = bytearray()
		
		for elem in self.iterelems():
			value = getattr(self, elem.tagname, None)
			
			if value is None:
				if elem.required:
					raise AttributeError(
					    "Missing required element \"{}\" in {}!".format(
					        elem.tagname,
					        self.__class__.__name__,
					    )
					)
			else:
				b += elem.asByteArray(value)
		
		return b
	
	def iterelems(self):
		for elem in self.__protobuf__:
			if isinstance(elem, tuple):
				for oneofElem in elem:
					yield oneofElem
			else:
				yield elem

class PebbleElement(object):
	"""Base class for a ProtoBuf message element."""
	
	def __init__(self, repeated, required, tagtype, tagname, tag, packed=True):
		self.repeated     = repeated
		self.required     = required
		self.tagtype      = tagtype
		self.tagname      = tagname
		self.tag          = tag
		self.packed       = packed
	
	def default(self):
		if   self.tagtype == "bool":   return False
		elif self.tagtype == "bytes":  return b""
		elif self.tagtype == "string": return ""
		else:                          return 0
	
	def asByteArray(self, data):
		required = self.required == "required"
		repeated = self.repeated == "repeated"
		packed   = self.packed
		b        = bytearray()
		
		if repeated:
			if packed and self.tagtype in ["double", "float", "int32", "int64",
			    "uint32", "uint64", "sint32", "sint64", "fixed32", "fixed64",
			    "sfixed32", "sfixed64", "bool", "enum"]:
				for datum in data:
					b += enc_value(self.tagtype, datum)
				b = enc_tagvalue("bytes", self.tag, b, required=required)
			else:
				for datum in data:
					b += enc_tagvalue(self.tagtype, self.tag, datum, required=True)
		else:
			b = enc_tagvalue(self.tagtype, self.tag, data, required=required)
		
		return b

#
# TF Protocol utility functions
#
# Mostly wire data encoders
#
def enc_uint64(i):
	i   = int(i)
	i  &= 0xFFFFFFFFFFFFFFFF
	b   = bytearray([i & 0x7F])
	i >>= 7
	
	while i:
		b[-1] |= 0x80
		b.append(i & 0x7F)
		i >>= 7
	
	return b
def enc_int64(i):    return enc_uint64(i)
def enc_uint32(i):   return enc_uint64(int(i) & 0xFFFFFFFF)
def enc_int32(i):    return enc_uint64(i)
def enc_sint64(i):
	i   = int(i)
	i  &= 0xFFFFFFFFFFFFFFFF
	i <<= 1
	i  ^= -(i>>64)
	i  &= 0xFFFFFFFFFFFFFFFF
	return enc_uint64(i)
def enc_sint32(i):
	i   = int(i)
	i  &= 0xFFFFFFFF
	i <<= 1
	i  ^= -(i>>32)
	i  &= 0xFFFFFFFF
	return enc_uint32(i)
def enc_bool(b):     return bytearray([1 if b else 0])
def enc_enum(e):     return enc_int32(e)
def enc_fixed64(i):  return bytearray(struct.pack("<Q", int  (i)))
def enc_sfixed64(i): return bytearray(struct.pack("<q", int  (i)))
def enc_float64(i):  return bytearray(struct.pack("<d", float(i)))
def enc_fixed32(i):  return bytearray(struct.pack("<I", int  (i)))
def enc_sfixed32(i): return bytearray(struct.pack("<i", int  (i)))
def enc_float32(i):  return bytearray(struct.pack("<f", float(i)))
def enc_string(i):   return bytearray(i, "utf-8") if isinstance(i, str) else i
def enc_tag(tag, wire):
	tag  = int(tag)
	wire = int(wire)
	assert tag >= 1                           and \
	       tag <= 2**29                       and \
	       not (tag >= 10000 and tag < 19999) and \
	       wire in [0,1,2,5]
	
	return enc_uint32(tag << 3 | wire)
def enc_delimited(buf):
	buf = bytearray(buf)
	return enc_uint64(len(buf)) + buf
def enc_value(tagtype, val):
	if   tagtype == "double":    return enc_float64(val)
	elif tagtype == "float":     return enc_float32(val)
	elif tagtype == "int32":     return enc_int32(val)
	elif tagtype == "int64":     return enc_int64(val)
	elif tagtype == "uint32":    return enc_uint32(val)
	elif tagtype == "uint64":    return enc_uint64(val)
	elif tagtype == "sint32":    return enc_sint32(val)
	elif tagtype == "sint64":    return enc_sint64(val)
	elif tagtype == "fixed32":   return enc_fixed32(val)
	elif tagtype == "fixed64":   return enc_fixed64(val)
	elif tagtype == "sfixed32":  return enc_sfixed32(val)
	elif tagtype == "sfixed64":  return enc_sfixed64(val)
	elif tagtype == "bool":      return enc_bool(val)
	elif tagtype == "enum":      return enc_enum(val)
	elif tagtype == "string":    return enc_string(val)
	elif tagtype == "bytes":     return bytearray(val)
	else:                        return val.asByteArray()
def enc_tagvalue(tagtype, tag, val, required=False):
	b = enc_value(tagtype, val)
	
	if   tagtype == "double":    return enc_tag(tag, 1)+b if(val or required) else bytearray()
	elif tagtype == "float":     return enc_tag(tag, 5)+b if(val or required) else bytearray()
	elif tagtype == "int32":     return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "int64":     return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "uint32":    return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "uint64":    return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "sint32":    return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "sint64":    return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "fixed32":   return enc_tag(tag, 5)+b if(val or required) else bytearray()
	elif tagtype == "fixed64":   return enc_tag(tag, 1)+b if(val or required) else bytearray()
	elif tagtype == "sfixed32":  return enc_tag(tag, 5)+b if(val or required) else bytearray()
	elif tagtype == "sfixed64":  return enc_tag(tag, 1)+b if(val or required) else bytearray()
	elif tagtype == "bool":      return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "enum":      return enc_tag(tag, 0)+b if(val or required) else bytearray()
	elif tagtype == "string" or \
	     tagtype == "bytes":
		b = enc_delimited(b);    return enc_tag(tag, 2)+b if(val or required) else bytearray()
	else:
		m = enc_delimited(b);    return enc_tag(tag, 2)+m if(b   or required) else bytearray()
