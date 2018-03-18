# -*- coding: utf-8 -*-

#
# Imports
#

import numpy     as np
import time

from   .pb   import *



__all__ = ["TfLogLevel", "TfSessionStatus", "TfDataType", "TfColorSpace",
           "TfGraphKeys", "TfEvent", "TfSummary", "TfLogMessage",
           "TfSessionLog", "TfTaggedRunMetadata", "TfValue",
           "TfSummaryMetadata", "TfPluginData", "TfImage", "TfHistogram",
           "TfAudio", "TfTensor", "TfTensorShape", "TfDim", "TfChart",
           "TfMultilineChart", "TfMarginChart", "TfMarginSeries", "TfCategory",
           "TfLayout", "HistogramAccumulator"]


#
# "Enums" with constants defined by TF
#

class TfLogLevel          (object):
	UNKNOWN      =  0
	DEBUGGING    = 10
	INFO         = 20
	WARN         = 30
	ERROR        = 40
	FATAL        = 50
	
	def __init__(self): raise

class TfSessionStatus     (object):
	UNSPECIFIED  =  0
	START        =  1
	STOP         =  2
	CHECKPOINT   =  3
	
	def __init__(self): raise

class TfDataType          (object):
	INVALID      =  0  # Not a legal value for DataType.  Used to indicate a DataType field has not been set.
	FLOAT        =  1  # Data types that all computation devices are expected to be
	DOUBLE       =  2  # capable to support.
	INT32        =  3
	UINT8        =  4
	INT16        =  5
	INT8         =  6
	STRING       =  7
	COMPLEX64    =  8  # Single-precision complex
	INT64        =  9
	BOOL         = 10
	QINT8        = 11  # Quantized int8
	QUINT8       = 12  # Quantized uint8
	QINT32       = 13  # Quantized int32
	BFLOAT16     = 14  # Float32 truncated to 16 bits.  Only for cast ops.
	QINT16       = 15  # Quantized int16
	QUINT16      = 16  # Quantized uint16
	UINT16       = 17
	COMPLEX128   = 18  # Double-precision complex
	HALF         = 19
	RESOURCE     = 20
	VARIANT      = 21  # Arbitrary C++ data types
	UINT32       = 22
	UINT64       = 23
	
	def __init__(self): raise

class TfColorSpace        (object):
	GRAYSCALE        = 1
	GRAYSCALE_ALPHA  = 2
	RGB              = 3
	RGBA             = 4
	DIGITAL_YUV      = 5
	BGRA             = 6
	
	def __init__(self): raise

class TfGraphKeys         (object):
	LOSSES                        = "losses"
	TRAINABLE_RESOURCE_VARIABLES  = "trainable_resource_variables"
	REGULARIZATION_LOSSES         = "regularization_losses"
	SAVERS                        = "savers"
	_SUMMARY_COLLECTION           = "_SUMMARY_V2"
	QUEUE_RUNNERS                 = "queue_runners"
	ACTIVATIONS                   = "activations"
	EVAL_STEP                     = "eval_step"
	GLOBAL_VARIABLES              = "variables"
	WHILE_CONTEXT                 = "while_context"
	TRAIN_OP                      = "train_op"
	GLOBAL_STEP                   = "global_step"
	SAVEABLE_OBJECTS              = "saveable_objects"
	METRIC_VARIABLES              = "metric_variables"
	READY_FOR_LOCAL_INIT_OP       = "ready_for_local_init_op"
	UPDATE_OPS                    = "update_ops"
	ASSET_FILEPATHS               = "asset_filepaths"
	WEIGHTS                       = "weights"
	INIT_OP                       = "init_op"
	LOCAL_INIT_OP                 = "local_init_op"
	LOCAL_RESOURCES               = "local_resources"
	TABLE_INITIALIZERS            = "table_initializer"
	_STREAMING_MODEL_PORTS        = "streaming_model_ports"
	BIASES                        = "biases"
	LOCAL_VARIABLES               = "local_variables"
	MOVING_AVERAGE_VARIABLES      = "moving_average_variables"
	RESOURCES                     = "resources"
	CONCATENATED_VARIABLES        = "concatenated_variables"
	TRAINABLE_VARIABLES           = "trainable_variables"
	READY_OP                      = "ready_op"
	COND_CONTEXT                  = "cond_context"
	SUMMARY_OP                    = "summary_op"
	MODEL_VARIABLES               = "model_variables"
	SUMMARIES                     = "summaries"
	
	def __init__(self): raise

#
# Message Hierarchy as defined by TF
#

class TfEvent             (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "double",              "wallTime",          1),
	PebbleElement("",         "",         "int64",               "step",              2),
	(
	PebbleElement("",         "",         "string",              "fileVersion",       3),
	PebbleElement("",         "",         "bytes",               "graphDef",          4),
	PebbleElement("",         "",         "TfSummary",           "summary",           5),
	PebbleElement("",         "",         "TfLogMessage",        "logMessage",        6),
	PebbleElement("",         "",         "TfSessionLog",        "sessionLog",        7),
	PebbleElement("",         "",         "TfTaggedRunMetadata", "taggedRunMetadata", 8),
	PebbleElement("",         "",         "bytes",               "metaGraphDef",      9),
	),
	]
	
	def __init__(self, step=0, wallTime=None, **kwargs):
		wallTime = time.time() if wallTime is None else float(wallTime)
		super().__init__(step=step, wallTime=wallTime, **kwargs)
	
	def asRecordByteArray(self):
		payload = self.asByteArray()
		header  = enc_fixed64(len(payload))
		
		return header                        + \
		       enc_fixed32(tfcrc32c(header)) + \
		       payload                       + \
		       enc_fixed32(tfcrc32c(payload))

class TfSummary           (PebbleMessage):
	__protobuf__ = [
	PebbleElement("repeated", "",         "TfValue",             "value",             1),
	]
	
	def asEvent(self, step=0, wallTime=None):
		return TfEvent(step, wallTime, summary=self)
	
	@classmethod
	def fromDict(kls, value={}):
		return TfSummary(value=[v for k,v in sorted(value.items())])

class TfLogMessage        (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "enum",              "level",             1),
	PebbleElement("",         "",         "string",            "message",           2),
	]
	
	def asEvent(self, step=0, wallTime=None):
		return TfEvent(step, wallTime, logMessage=self)

class TfSessionLog        (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "enum",              "status",            1),
	PebbleElement("",         "",         "string",            "checkpointPath",    2),
	PebbleElement("",         "",         "string",            "message",           3),
	]
	
	def asEvent(self, step=0, wallTime=None):
		return TfEvent(step, wallTime, sessionLog=self)

class TfTaggedRunMetadata (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "string",            "tag",               1),
	PebbleElement("",         "",         "bytes",             "runMetadata",       2),
	]
	
	def asEvent(self, step=0, wallTime=None):
		return TfEvent(step, wallTime, taggedRunMetadata=self)

class TfValue             (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "string",              "tag",               1),
	PebbleElement("",         "",         "TfSummaryMetadata",   "metadata",          9),
	(
	PebbleElement("",         "",         "float",               "simpleValue",       2),
	PebbleElement("",         "",         "TfImage",             "image",             4),
	PebbleElement("",         "",         "TfHistogram",         "histo",             5),
	PebbleElement("",         "",         "TfAudio",             "audio",             6),
	PebbleElement("",         "",         "TfTensor",            "tensor",            8),
	),
	]

class TfSummaryMetadata   (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "TfPluginData",        "pluginData",        1),
	PebbleElement("",         "",         "string",              "displayName",       2),
	PebbleElement("",         "",         "string",              "description",       3),
	]

class TfPluginData        (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "string",            "pluginName",        1),
	PebbleElement("",         "",         "bytes",             "content",           2),
	]

class TfImage             (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "int32",             "height",            1),
	PebbleElement("",         "",         "int32",             "width",             2),
	PebbleElement("",         "",         "int32",             "colorspace",        3),
	PebbleElement("",         "",         "bytes",             "imageData",         4),
	]
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag=tag, metadata=metadata, image=self)

class TfHistogram         (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "double",            "min",               1),
	PebbleElement("",         "",         "double",            "max",               2),
	PebbleElement("",         "",         "double",            "num",               3),
	PebbleElement("",         "",         "double",            "sum",               4),
	PebbleElement("",         "",         "double",            "sumSquares",        5),
	PebbleElement("repeated", "",         "double",            "bucketLimits",      6),
	PebbleElement("repeated", "",         "double",            "buckets",           7),
	]
	
	@classmethod
	def fromNumpy(self, *tensors, bucketLimits=None):
		hacc = HistogramAccumulator(*tensors, bucketLimits=bucketLimits)
		return TfHistogram(**hacc)
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag=tag, metadata=metadata, histo=self)

class TfAudio             (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "float",             "sampleRate",        1),
	PebbleElement("",         "",         "int64",             "numChannels",       2),
	PebbleElement("",         "",         "int64",             "lengthFrames",      3),
	PebbleElement("",         "",         "bytes",             "audioData",         4),
	PebbleElement("",         "",         "string",            "contentType",       5),
	]
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag=tag, metadata=metadata, audio=self)

class TfTensor            (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "enum",                "dtype",             1),
	PebbleElement("",         "",         "TfTensorShape",       "tensorShape",       2),
	PebbleElement("",         "",         "int32",               "versionNumber",     3),
	PebbleElement("",         "",         "bytes",               "tensorVal",         4),
	PebbleElement("repeated", "",         "bytes",               "stringVal",         8),
	]
	
	@classmethod
	def fromText(kls, t, dimNames=None):
		dtype, shape, stringVal = tensorFromText (t, dimNames)
		return TfTensor(dtype=dtype, tensorShape=shape, stringVal=stringVal)
	
	@classmethod
	def fromNumpy(kls, t, dimNames=None):
		dtype, shape, tensorVal = tensorFromNumpy(t, dimNames)
		return TfTensor(dtype=dtype, tensorShape=shape, tensorVal=tensorVal)
	
	def asValue(self, tag, metadata=None):
		return TfValue(tag=tag, metadata=metadata, tensor=self)

class TfTensorShape       (PebbleMessage):
	__protobuf__ = [
	PebbleElement("repeated", "",         "TfDim",               "dim",               2),
	PebbleElement("",         "",         "bool",                "unknownRank",       3),
	]

class TfDim               (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "int64",             "size",              1),
	PebbleElement("",         "",         "string",            "name",              2),
	]

class TfChart             (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "string",              "title",             1),
	(
	PebbleElement("",         "",         "TfMultilineChart",    "multiline",         2),
	PebbleElement("",         "",         "TfMarginChart",       "margin",            3),
	),
	]

class TfMultilineChart    (PebbleMessage):
	__protobuf__ = [
	PebbleElement("repeated", "",         "string",            "tag",               1),
	]

class TfMarginChart       (PebbleMessage):
	__protobuf__ = [
	PebbleElement("repeated", "",         "TfMarginSeries",      "series",            1),
	]

class TfMarginSeries      (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "string",            "value",             1),
	PebbleElement("",         "",         "string",            "lower",             2),
	PebbleElement("",         "",         "string",            "upper",             3),
	]

class TfCategory          (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "string",              "title",             1),
	PebbleElement("repeated", "",         "TfChart",             "chart",             2),
	PebbleElement("",         "",         "bool",                "closed",            3),
	]

class TfLayout            (PebbleMessage):
	__protobuf__ = [
	PebbleElement("",         "",         "int32",               "version",           1),
	PebbleElement("repeated", "",         "TfCategory",          "category",          2),
	]


#
# Utilities
#
class HistogramAccumulator(object):
	def __init__(self, *args, bucketLimits=None):
		self.min          = np.nan
		self.max          = np.nan
		self.num          = 0
		self.sum          = 0.0
		self.sumSquares   = 0.0
		self.bucketLimits = np.array(self.getDefaultBucketLimits()
		                             if bucketLimits is None else
		                             bucketLimits, dtype="float64")
		self.buckets      = np.zeros_like(self.bucketLimits, dtype="int64")
		
		for a in args:
			self += a
	
	def __iadd__(self, h):
		if isinstance(h, HistogramAccumulator):
			assert np.alltrue(self.bucketLimits == h.bucketLimits)
			self.min         = self.safemin(self.min, h.min)
			self.max         = self.safemax(self.max, h.max)
			self.num        += h.num
			self.sum        += h.sum
			self.sumSquares += h.sumSquares
			self.buckets    += h.buckets
		else:
			self.min         = self.safemin(self.min, float(np.nanmin(h)))
			self.max         = self.safemax(self.max, float(np.nanmax(h)))
			self.num        += int(np.prod(h.shape))
			self.sum        += float(np.sum(h.astype("float64")))
			self.sumSquares += float(np.sum(h.astype("float64")**2))
			self.buckets    += np.histogram(h, [np.finfo("float64").min] +
			                                   list(self.bucketLimits))[0]
		
		return self
	
	@classmethod
	def safemax(self, a, b):
		a = float(a)
		b = float(b)
		if   np.isnan(a):
			return b
		elif np.isnan(b):
			return a
		elif b>a:
			return b
		else:
			return a
	
	@classmethod
	def safemin(self, a, b):
		a = float(a)
		b = float(b)
		if   np.isnan(a):
			return b
		elif np.isnan(b):
			return a
		elif b<a:
			return b
		else:
			return a
	
	def __getitem__(self, key):
		if key in self.keys():
			return self.__dict__[key]
		else:
			raise KeyError("Invalid key!")
	
	def keys(self):
		return {"min", "max", "num", "sum", "sumSquares", "bucketLimits", "buckets"}
	
	@classmethod
	def getDefaultBucketLimits(kls):
		"""
		Compute the default histogram buckets used by TF.
		"""
		buckets = []
		lim     = 1e-12
		while lim<1e20:
			buckets += [lim]
			lim     *= 1.1
		return [-l for l in buckets[::-1]] + [0] + buckets + [np.finfo("float64").max]


def getTextTensorShape(t):
	#
	# Compute the shape of the tensor, avoiding problems like lists containing
	# themselves and other traps by memoizing the sublists we've entered.
	#
	# On the assumption that this tensor is rectangular,
	#   - len(t[0])        gives the length of dimension 0
	#   - len(t[0][0])     gives the length of dimension 1
	#   - len(t[0][0][0])  gives the length of dimension 2
	#   - ...
	# and this continues until we hit a bytes/bytearray/str object.
	#
	memo = []
	curr = t
	while not isinstance(curr, (bytes, bytearray, str)):
		if curr in memo:
			raise ValueError("Infinite loop detected!")
		if len(memo) >= 32:
			raise ValueError("Excessively-high-rank tensor detected!")
		if len(curr) == 0:
			raise ValueError("Zero-length dimension detected!")
		memo += [curr]
		curr  = curr[0]
	
	return tuple(map(len, memo))

def flatiter(t, l, s):
	#
	# Iterate over tensor, returning its elements in row-major (C) order
	#
	# Throughout, check that the list of lists represents a rectangular, and
	# not ragged, array.
	#
	if len(t) != l:
		raise ValueError("Ragged array detected!")
	
	if not s:
		yield from t
	else:
		for tt in t:
			yield from flatiter(tt, s[0], s[1:])

def tensorFromText(t, dimNames=None):
	#
	# Determine the shape of the tensor. Then, use knowledge of the shape to
	# flatten it into an array of bytearray().
	#
	def byteify(s):
		return bytearray(s, "utf-8") if isinstance(s, str) else bytearray(s)
	
	shape       = getTextTensorShape(t)
	tensorShape = convert_shape_np2tf(shape, dimNames)
	stringVal   = [byteify(s) for s in flatiter([t], 1, shape)]
	return TfDataType.STRING, tensorShape, stringVal

def tensorFromNumpy(t, dimNames=None):
	t         = np.array(t)
	dtype     = convert_dtype_np2tf(t.dtype)
	shape     = convert_shape_np2tf(t.shape, dimNames)
	tensorVal = t.tobytes('C')
	return dtype, shape, tensorVal

def convert_dtype_np2tf(dtype):
	"""Convert a numpy array's dtype to the enum DataType code used by TF."""
	
	dtype  = str(dtype)
	dtype  = {
		"invalid":     TfDataType.INVALID,
		"float32":     TfDataType.FLOAT,
		"float64":     TfDataType.DOUBLE,
		"int32":       TfDataType.INT32,
		"uint8":       TfDataType.UINT8,
		"int16":       TfDataType.INT16,
		"int8":        TfDataType.INT8,
		"string":      TfDataType.STRING,
		"complex64":   TfDataType.COMPLEX64,
		"int64":       TfDataType.INT64,
		"bool":        TfDataType.BOOL,
		# 11-16: Quantized datatypes that don't exist in numpy...
		"uint16":      TfDataType.UINT16,
		"complex128":  TfDataType.COMPLEX128,
		"float16":     TfDataType.HALF,
		# 20-21: TF-specific datatypes...
		"uint32":      TfDataType.UINT32,
		"uint64":      TfDataType.UINT64,
	}[dtype]
	return dtype

def convert_dims_np2tf (shape, dimNames=None):
	if dimNames is None:
		return [TfDim(size=size) for size in shape]
	else:
		assert len(dimNames) == len(shape)
		dimNames = [str(x) for x in dimNames]
		return [TfDim(size=size, name=name) for size, name in zip(shape, dimNames)]

def convert_shape_np2tf(shape, dimNames=None):
	return TfTensorShape(dim=convert_dims_np2tf(shape, dimNames))

def convert_metadata   (displayName = None,
                        description = None,
                        pluginName  = None,
                        content     = None):
	if pluginName  is None and content is None:
		pluginData = None
	else:
		pluginData = TfPluginData(pluginName=pluginName, content=content)
	
	if displayName is None and description is None and pluginData is None:
		metadata   = None
	else:
		metadata   = TfSummaryMetadata(
		                 displayName = displayName,
		                 description = description,
		                 pluginData  = pluginData
		             )
	
	return metadata


"""
A tiny module to emit TFEvents files.

TFEvents files are a concatenation of "records":

	Record:
		uint64_t dataLen                // Little-Endian
		uint32_t dataLen_maskCRC32C     // Little-Endian
		uint8_t  data[dataLen]
		uint32_t data_maskCRC32C        // Little-Endian
	(repeat unlimited number of times)


The masked CRC32C is defined in Java as

	private static long maskedCRC32(byte[] data){
		crc32.reset();
		crc32.update(data, 0, data.length);
		long x = u32(crc32.getValue());
		return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8);
	}

where the CRC32C is initialized at ~0, uses the Castagnoli polynomial
and is finalized by bit-reversal.


The payload of a Record is a single protobuf Event, defined with all
submessage below (shamelessly ripped off from TensorFlow Github). The first
Record of a file must always contain Event(file_Version="brain.Event:2")


The TFEvents protocol is defined by the contents of the TensorFlow .proto files
below, organized by the hierarchy of their inclusion, namely:

	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto
		https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
			https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_handle.proto
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto
	https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/custom_scalar/layout.proto

The entities that exist in those files are below, organized by the hierarchy of
their inclusion (irrelevant/obsolete members removed):

	Event
		double      wall_time
		int64       step
		oneof:
			string      file_version
			LogMessage  log_message
				enum        level
					UNKNOWN:     0
					DEBUGGING:  10
					INFO:       20
					WARN:       30
					ERROR:      40
					FATAL:      50
				string      message
			SessionLog  session_log
				enum        status
					UNSPECIFIED: 0
					START:       1
					STOP:        2
					CHECKPOINT:  3
				string      checkpoint_path
				string      msg
			Summary     summary
				repeated Value value
					string           tag
					SummaryMetadata  metadata
						PluginData
							string plugin_name
							bytes  content
						string display_name
						string summary_description
					float            simple_value
					Image            image
						int   height
						int   width
						int   colorspace
							grayscale:        1
							grayscale+alpha:  2
							RGB:              3
							RGBA:             4
							DIGITAL_YUV:      5
							BGRA:             6
						bytes data
					HistogramProto   histo
						double           min
						double           max
						double           num
						double           sum
						repeated packed double sum_squares
						repeated packed double bucket_limit
					Audio            audio
						float            sample_rate   // In Hz
						int              num_channels
						int              length_franes
						bytes            data
						string           content_type
					TensorProto      tensor
						enum DataType    dtype
							DT_INVALID = 0;// Not a legal value for DataType.  Used to indicate a DataType field has not been set.
							DT_FLOAT = 1;  // Data types that all computation devices are expected to be
							DT_DOUBLE = 2; // capable to support.
							DT_INT32 = 3;
							DT_UINT8 = 4;
							DT_INT16 = 5;
							DT_INT8 = 6;
							DT_STRING = 7;
							DT_COMPLEX64 = 8;  // Single-precision complex
							DT_INT64 = 9;
							DT_BOOL = 10;
							DT_QINT8 = 11;     // Quantized int8
							DT_QUINT8 = 12;    // Quantized uint8
							DT_QINT32 = 13;    // Quantized int32
							DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
							DT_QINT16 = 15;    // Quantized int16
							DT_QUINT16 = 16;   // Quantized uint16
							DT_UINT16 = 17;
							DT_COMPLEX128 = 18;  // Double-precision complex
							DT_HALF = 19;
							DT_RESOURCE = 20;
							DT_VARIANT = 21;  // Arbitrary C++ data types
							DT_UINT32 = 22;
							DT_UINT64 = 23;
						TensorShapeProto tensor_shape
							repeated Dim dim
								int    size
								string name
							bool unknown_rank // == 0 for all cases of concern to us
						int          version_number   // == 0
						bytes        tensor_content   // Row-major (C-contiguous) order
	
	# Custom Scalars plugin (custom_scalars):
	Layout            layout
		int32             version
		repeated Category category
			string            title
			bool              closed
			repeated Chart    chart
				string            title
				repeated string   tag







// Protocol buffer representing an event that happened during
// the execution of a Brain model.
message Event {
  // Timestamp of the event.
  double wall_time = 1;

  // Global step of the event.
  int64 step = 2;

  oneof what {
    // An event file was started, with the specified version.
    // This is use to identify the contents of the record IO files
    // easily.  Current version is "brain.Event:2".  All versions
    // start with "brain.Event:".
    string file_version = 3;
    // An encoded version of a GraphDef.
    bytes graph_def = 4;
    // A summary was generated.
    Summary summary = 5;
    // The user output a log message. Not all messages are logged, only ones
    // generated via the Python tensorboard_logging module.
    LogMessage log_message = 6;
    // The state of the session which can be used for restarting after crashes.
    SessionLog session_log = 7;
    // The metadata returned by running a session.run() call.
    TaggedRunMetadata tagged_run_metadata = 8;
    // An encoded version of a MetaGraphDef.
    bytes meta_graph_def = 9;
  }
}

// Protocol buffer used for logging messages to the events file.
message LogMessage {
  enum Level {
    UNKNOWN = 0;
    // Note: The logging level 10 cannot be named DEBUG. Some software
    // projects compile their C/C++ code with -DDEBUG in debug builds. So the
    // C++ code generated from this file should not have an identifier named
    // DEBUG.
    DEBUGGING = 10;
    INFO = 20;
    WARN = 30;
    ERROR = 40;
    FATAL = 50;
  }
  Level level = 1;
  string message = 2;
}

// Protocol buffer used for logging session state.
message SessionLog {
  enum SessionStatus {
    STATUS_UNSPECIFIED = 0;
    START = 1;
    STOP = 2;
    CHECKPOINT = 3;
  }

  SessionStatus status = 1;
  // This checkpoint_path contains both the path and filename.
  string checkpoint_path = 2;
  string msg = 3;
}

// For logging the metadata output for a single session.run() call.
message TaggedRunMetadata {
  // Tag name associated with this metadata.
  string tag = 1;
  // Byte-encoded version of the `RunMetadata` proto in order to allow lazy
  // deserialization.
  bytes run_metadata = 2;
}


// Metadata associated with a series of Summary data
message SummaryDescription {
  // Hint on how plugins should process the data in this series.
  // Supported values include "scalar", "histogram", "image", "audio"
  string type_hint = 1;
}

// Serialization format for histogram module in
// core/lib/histogram/histogram.h
message HistogramProto {
  double min = 1;
  double max = 2;
  double num = 3;
  double sum = 4;
  double sum_squares = 5;

  // Parallel arrays encoding the bucket boundaries and the bucket values.
  // bucket(i) is the count for the bucket i.  The range for
  // a bucket is:
  //   i == 0:  -DBL_MAX .. bucket_limit(0)
  //   i != 0:  bucket_limit(i-1) .. bucket_limit(i)
  repeated double bucket_limit = 6 [packed = true];
  repeated double bucket = 7 [packed = true];
};

// A SummaryMetadata encapsulates information on which plugins are able to make
// use of a certain summary value.
message SummaryMetadata {
  message PluginData {
    // The name of the plugin this data pertains to.
    string plugin_name = 1;

    // The content to store for the plugin. The best practice is for this to be
    // a binary serialized protocol buffer.
    bytes content = 2;
  }

  // Data that associates a summary with a certain plugin.
  PluginData plugin_data = 1;

  // Display name for viewing in TensorBoard.
  string display_name = 2;

  // Longform readable description of the summary sequence. Markdown supported.
  string summary_description = 3;
};

// A Summary is a set of named values to be displayed by the
// visualizer.
//
// Summaries are produced regularly during training, as controlled by
// the "summary_interval_secs" attribute of the training operation.
// Summaries are also produced at the end of an evaluation.
message Summary {
  message Image {
    // Dimensions of the image.
    int32 height = 1;
    int32 width = 2;
    // Valid colorspace values are
    //   1 - grayscale
    //   2 - grayscale + alpha
    //   3 - RGB
    //   4 - RGBA
    //   5 - DIGITAL_YUV
    //   6 - BGRA
    int32 colorspace = 3;
    // Image data in encoded format.  All image formats supported by
    // image_codec::CoderUtil can be stored here.
    bytes encoded_image_string = 4;
  }

  message Audio {
    // Sample rate of the audio in Hz.
    float sample_rate = 1;
    // Number of channels of audio.
    int64 num_channels = 2;
    // Length of the audio in frames (samples per channel).
    int64 length_frames = 3;
    // Encoded audio data and its associated RFC 2045 content type (e.g.
    // "audio/wav").
    bytes encoded_audio_string = 4;
    string content_type = 5;
  }

  message Value {
    // This field is deprecated and will not be set.
    string node_name = 7;

    // Tag name for the data. Used by TensorBoard plugins to organize data. Tags
    // are often organized by scope (which contains slashes to convey
    // hierarchy). For example: foo/bar/0
    string tag = 1;

    // Contains metadata on the summary value such as which plugins may use it.
    // Take note that many summary values may lack a metadata field. This is
    // because the FileWriter only keeps a metadata object on the first summary
    // value with a certain tag for each tag. TensorBoard then remembers which
    // tags are associated with which plugins. This saves space.
    SummaryMetadata metadata = 9;

    // Value associated with the tag.
    oneof value {
      float simple_value = 2;
      bytes obsolete_old_style_histogram = 3;
      Image image = 4;
      HistogramProto histo = 5;
      Audio audio = 6;
      TensorProto tensor = 8;
    }
  }

  // Set of values for the summary.
  repeated Value value = 1;
}

// Protocol buffer representing a tensor.
message TensorProto {
  DataType dtype = 1;

  // Shape of the tensor.  TODO(touts): sort out the 0-rank issues.
  TensorShapeProto tensor_shape = 2;

  // Only one of the representations below is set, one of "tensor_contents" and
  // the "xxx_val" attributes.  We are not using oneof because as oneofs cannot
  // contain repeated fields it would require another extra set of messages.

  // Version number.
  //
  // In version 0, if the "repeated xxx" representations contain only one
  // element, that element is repeated to fill the shape.  This makes it easy
  // to represent a constant Tensor with a single value.
  int32 version_number = 3;

  // Serialized raw tensor content from either Tensor::AsProtoTensorContent or
  // memcpy in tensorflow::grpc::EncodeTensorToByteBuffer. This representation
  // can be used for all tensor types. The purpose of this representation is to
  // reduce serialization overhead during RPC call by avoiding serialization of
  // many repeated small items.
  bytes tensor_content = 4;

  // Type specific representations that make it easy to create tensor protos in
  // all languages.  Only the representation corresponding to "dtype" can
  // be set.  The values hold the flattened representation of the tensor in
  // row major order.

  // DT_HALF. Note that since protobuf has no int16 type, we'll have some
  // pointless zero padding for each value here.
  repeated int32 half_val = 13 [packed = true];

  // DT_FLOAT.
  repeated float float_val = 5 [packed = true];

  // DT_DOUBLE.
  repeated double double_val = 6 [packed = true];

  // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
  repeated int32 int_val = 7 [packed = true];

  // DT_STRING
  repeated bytes string_val = 8;

  // DT_COMPLEX64. scomplex_val(2*i) and scomplex_val(2*i+1) are real
  // and imaginary parts of i-th single precision complex.
  repeated float scomplex_val = 9 [packed = true];

  // DT_INT64
  repeated int64 int64_val = 10 [packed = true];

  // DT_BOOL
  repeated bool bool_val = 11 [packed = true];

  // DT_COMPLEX128. dcomplex_val(2*i) and dcomplex_val(2*i+1) are real
  // and imaginary parts of i-th double precision complex.
  repeated double dcomplex_val = 12 [packed = true];

  // DT_RESOURCE
  repeated ResourceHandleProto resource_handle_val = 14;

  // DT_VARIANT
  repeated VariantTensorDataProto variant_val = 15;

  // DT_UINT32
  repeated uint32 uint32_val = 16 [packed = true];

  // DT_UINT64
  repeated uint64 uint64_val = 17 [packed = true];
};

// Protocol buffer representing the serialization format of DT_VARIANT tensors.
message VariantTensorDataProto {
  // Name of the type of objects being serialized.
  string type_name = 1;
  // Portions of the object that are not Tensors.
  bytes metadata = 2;
  // Tensors contained within objects being serialized.
  repeated TensorProto tensors = 3;
}

// Protocol buffer representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
message ResourceHandleProto {
  // Unique name for the device containing the resource.
  string device = 1;

  // Container in which this resource is placed.
  string container = 2;

  // Unique name of this resource.
  string name = 3;

  // Hash code for the type of the resource. Is only valid in the same device
  // and in the same execution.
  uint64 hash_code = 4;

  // For debug-only, the name of the type pointed to by this handle, if
  // available.
  string maybe_type_name = 5;
};

// Dimensions of a tensor.
message TensorShapeProto {
  // One dimension of the tensor.
  message Dim {
    // Size of the tensor in that dimension.
    // This value must be >= -1, but values of -1 are reserved for "unknown"
    // shapes (values of -1 mean "unknown" dimension).  Certain wrappers
    // that work with TensorShapeProto may fail at runtime when deserializing
    // a TensorShapeProto containing a dim value of -1.
    int64 size = 1;

    // Optional name of the tensor dimension.
    string name = 2;
  };

  // Dimensions of the tensor, such as {"input", 30}, {"output", 40}
  // for a 30 x 40 2D tensor.  If an entry has size -1, this
  // corresponds to a dimension of unknown size. The names are
  // optional.
  //
  // The order of entries in "dim" matters: It indicates the layout of the
  // values in the tensor in-memory representation.
  //
  // The first entry in "dim" is the outermost dimension used to layout the
  // values, the last entry is the innermost dimension.  This matches the
  // in-memory layout of RowMajor Eigen tensors.
  //
  // If "dim.size()" > 0, "unknown_rank" must be false.
  repeated Dim dim = 2;

  // If true, the number of dimensions in the shape is unknown.
  //
  // If true, "dim.size()" must be 0.
  bool unknown_rank = 3;
};

// LINT.IfChange
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types
  DT_UINT32 = 22;
  DT_UINT64 = 23;

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF = 101;
  DT_DOUBLE_REF = 102;
  DT_INT32_REF = 103;
  DT_UINT8_REF = 104;
  DT_INT16_REF = 105;
  DT_INT8_REF = 106;
  DT_STRING_REF = 107;
  DT_COMPLEX64_REF = 108;
  DT_INT64_REF = 109;
  DT_BOOL_REF = 110;
  DT_QINT8_REF = 111;
  DT_QUINT8_REF = 112;
  DT_QINT32_REF = 113;
  DT_BFLOAT16_REF = 114;
  DT_QINT16_REF = 115;
  DT_QUINT16_REF = 116;
  DT_UINT16_REF = 117;
  DT_COMPLEX128_REF = 118;
  DT_HALF_REF = 119;
  DT_RESOURCE_REF = 120;
  DT_VARIANT_REF = 121;
  DT_UINT32_REF = 122;
  DT_UINT64_REF = 123;
}

/**
 * Encapsulates information on a single chart. Many charts appear in a category.
 */
message Chart {
  // The title shown atop this chart. Optional. Defaults to 'untitled'.
  string title = 1;

  // The content of the chart. This depends on the type of the chart.
  oneof content {
    MultilineChartContent multiline = 2;
    MarginChartContent margin = 3;
  }
}

/**
 * Encapsulates information on a single line chart. This line chart may have
 * lines associated with several tags.
 */
message MultilineChartContent {
  // A list of regular expressions for tags that should appear in this chart.
  // Tags are matched from beginning to end. Each regex captures a set of tags.
  repeated string tag = 1;
}

/**
 * Encapsulates information on a single margin chart. A margin chart uses fill
 * area to visualize lower and upper bounds that surround a value.
 */
message MarginChartContent {
  /**
   * Encapsulates a tag of data for the chart.
   */
  message Series {
    // The exact tag string associated with the scalar summaries making up the
    // main value between the bounds.
    string value = 1;

    // The exact tag string associated with the scalar summaries making up the
    // lower bound.
    string lower = 2;

    // The exact tag string associated with the scalar summaries making up the
    // upper bound.
    string upper = 3;
  }

  // A list of data series to include within this margin chart.
  repeated Series series = 1;
}

/**
 * A category contains a group of charts. Each category maps to a collapsible
 * within the dashboard.
 */
message Category {
  // This string appears atop each grouping of charts within the dashboard.
  string title = 1;

  // Encapsulates data on charts to be shown in the category.
  repeated Chart chart = 2;

  // Whether this category should be initially closed. False by default.
  bool closed = 3;
}

/**
 * A layout encapsulates how charts are laid out within the custom scalars
 * dashboard.
 */
message Layout {
  // Version `0` is the only supported version.
  int32 version = 1;

  // The categories here are rendered from top to bottom.
  repeated Category category = 2;
}








TO BE LOOKED AT:
	https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/projector/projector_config.proto
	https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve
"""
