# -*- coding: utf-8 -*-

#
# Imports
#

import                        contextlib
import                        os
import                        pickle       as pkl
import                        numpy        as np
import                        re
import                        sys
import                        threading
import                        time
import                        uuid

from   io              import BytesIO

from   .tfevents       import *
from   .tfevents       import convert_metadata

__all__ = [
    "ZvitWriter", "NullZvitWriter", "tagscope", "topZvit",
    "logScalar", "logScalars", "logImage",   "logAudio",   "logText",
    "logTensor", "logHist",    "logMessage", "logSession", "logLayout",
    "top_zvit",    "log_scalar",  "log_scalars", "log_image",
    "log_audio",   "log_text",    "log_tensor",  "log_hist",
    "log_message", "log_session", "log_layout",
    "TfLogLevel", "TfSessionStatus", "TfDataType", "TfColorSpace",
    "TfGraphKeys", "TfEvent", "TfSummary", "TfLogMessage",
    "TfSessionLog", "TfTaggedRunMetadata", "TfValue",
    "TfSummaryMetadata", "TfPluginData", "TfImage", "TfHistogram",
    "TfAudio", "TfTensor", "TfTensorShape", "TfDim", "TfChart",
    "TfMultilineChart", "TfMarginChart", "TfMarginSeries", "TfCategory",
    "TfLayout", "HistogramAccumulator",
]


#
# Zvit Writer.
#

class ZvitWriter(object):
	"""
	ZvitWriter is a class for event logging to Protocol Buffer files
	compatible with TensorBoard.
	"""
	
	def __init__            (self, logDir, step,
	                               flushSecs         = 5.0,
	                               flushBufSz        = None,
	                               tagMatcher        = None,
	                               collectionMatcher = "^"+TfGraphKeys.SUMMARIES+"$"):
		"""
		ZvitWriter
		
		logDir:            Directory in which the logfiles will be created.
		step:              Initial value of the global step # to begin at.
		flushSecs:         Flush the internal buffer asynchronously every given # of
		                   seconds. <= 0.0 indicates no asynchronous flushing.
		flushBufSz:        Flush the internal buffer synchronously when it exceeds
		                   the given size in bytes. <= 0 bytes indicates no maximum.
		tagMatcher:        Filter summaries by tag. Can be a regex or callable.
		collectionMatcher: Filter summaries by collection. Can be a regex or callable.
		"""
		
		#
		# Training logistics-related
		#
		self._tagMatcher        = re.compile(tagMatcher)       .match if isinstance(tagMatcher,        str) else tagMatcher
		self._collectionMatcher = re.compile(collectionMatcher).match if isinstance(collectionMatcher, str) else collectionMatcher
		self._logDir            = str(logDir)
		self._globalStep        = int(step)
		self._logFileTime       = time.time()
		self._uuid              = uuid.uuid4()
		self._MB                = set()              # Metadata Buffer
		self._VB                = dict()             # Value    Buffer
		self._BB                = bytearray()        # Byte     Buffer
		self._FH                = bytearray()        # File     Header
		
		#
		# Threading-related
		#
		self._flushSecs         = flushSecs
		self._flushBufSz        = flushBufSz
		self._flushThread       = None
		self._lock              = threading.Condition()
		self._tls               = threading.local()
		
		#
		# Compute the FH (file header) of sorts, consisting of the fileVersion
		# event and TfSessionLog event.
		#
		# We do *NOT* append the FH to the BB, nor flush the BB afterwards. The
		# FH is lazily prepended to the BB the first time that data is appended
		# to the BB.
		#
		# We want to delay file creation for as long as possible. This is
		# because if the program crashes before the ZvitWriter can be used,
		# or the writer is not used before being deleted or crashing, we
		# would like it not to create embryonic, empty tfevents files, thus
		# polluting the filesystem and slowing TensorBoard down.
		#
		# Thus by withholding the FH from the BB until other data is about to
		# be appended, a flush() such as that in __del__() will have no effect.
		#
		self._FH += TfEvent(
		                self.globalStep,
		                self.logFileTime,
		                fileVersion="brain.Event:2"
		            ).asRecordByteArray()
		self._FH += TfSessionLog(
		                status         = TfSessionStatus.START,
		                checkpointPath = self.logDir,
		                message        = "Restarting...",
		            ).asEvent(
		                self.globalStep,
		                self.logFileTime
		            ).asRecordByteArray()
		
		#
		# With the ZvitWriter mostly initialized and possessing all of the
		# fields it should, assert the existence of the log directory and
		# nonexistence of the file we're about to create.
		#
		self._initAsserts()
	
	def __repr__            (self):
		return "{}({}, {}, {}, {}, {}, {})".format(
		    repr(self.__class__.__name__),
		    repr(self._logDir),
		    repr(self._globalStep),
		    repr(self._flushSecs),
		    repr(self._flushBufSz),
		    repr(self._tagMatcher),
		    repr(self._collectionMatcher),
		)
	
	def __del__             (self):
		self.close()
		getattr(super(), "__del__", lambda:None)()
	
	def __enter__           (self):
		"""
		Make this zvit writer the default writer for the current thread.
		
		This is done by pushing it onto the stack of writers.
		"""
		
		return self.push()
	def __exit__            (self, *exc):
		"""
		Remove this zvit writer as the default writer for the current thread.
		
		This is done by popping it from the stack of writers on context exit.
		"""
		
		popped = self.close().pop()
		assert popped == self
	
	def __getstate__        (self):
		self.flush()
		return (self._logDir,
		        self._globalStep,
		        self._flushSecs,
		        self._flushBufSz,
		        self._tagMatcher,
		        self._collectionMatcher,)
	
	def __setstate__        (self, args):
		self.__init__(*args)
	
	
	#
	# Fundamental, readonly properties.
	#
	@property
	def logDir              (self):
		"""
		Return log directory.
		"""
		
		return self._logDir
	
	@property
	def startStep           (self):
		return self._startStep
	
	@property
	def globalStep          (self):
		"""
		Current global step number of logging.
		"""
		return self._globalStep
	
	@property
	def logFileName         (self):
		"""
		Filename this ZvitWriter will log to.
		
		A name made of a zero-padded, 30-digit, nanosecond-resolution POSIX
		time plus a UUID4 in that order is
		
		    1) Constant-length for the next 100 million years
		    2) Unique with extremely high probability even when one of the
		       entropy sources (time or RNG) is broken, but not when both are
		       (e.g. when RNG is based on time)
		    3) Lexicographically sorts after any previously-created name,
		       provided the wallclock time increases monotonically.
		"""
		
		logFileName = "tfevents-{:030.9f}-{:s}-GUID-{:s}.zvit".format(
		    self.logFileTime,
		    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.logFileTime)),
		    str(self.uuid),
		)
		return logFileName
	
	@property
	def logFilePath         (self):
		"""
		Full path to file this ZvitWriter will log to.
		"""
		
		return os.path.join(self.logDir, self.logFileName)
	
	@property
	def logFileTime         (self):
		"""
		Timestamp of the earliest possible event in the file.
		
		This timestamp is also incorporated into the filename.
		"""
		
		return self._logFileTime
	
	@property
	def uuid                (self):
		return self._uuid
	
	@property
	def asynchronous        (self):
		return (self._flushSecs  is not None) and (self._flushSecs        > 0.0)
	
	@property
	def overflowing         (self):
		return (self._flushBufSz is not None) and (len(self._BB) > self._flushBufSz)
	
	@property
	def _tagScopeStack      (self):
		"""
		Returns the *current thread's* nested tag scopes *only*.
		
		In particular, a debugger thread *will not* see the same values as the
		debuggee thread.
		"""
		
		return self._tls.__dict__.setdefault("tagScopeStack", [])
	
	
	#
	# Context managers.
	#
	@contextlib.contextmanager
	def tagscope            (self, *groupNames):
		"""
		Enter a named tag scope in the current thread.
		
		The tag subgroup names may not contain the "/" hierarchical separator.
		"""
		
		namesPushed = 0
		for groupName in groupNames:
			try:
				self.pushTag(groupName)
				namesPushed += 1
			except:
				# Unwind the names partially stacked, then repropagate exception
				exc_info = sys.exc_info()
				for groupName in groupNames[:namesPushed:-1]:
					self.popTag()
				raise exc_info
		
		yield self
		
		for groupName in groupNames:
			self.popTag()
	
	
	#
	# Internals. Do not touch.
	#
	def _initAsserts        (self):
		"""
		Internal check that the log directory we want does exist but that the
		log file does not.
		"""
		
		#
		# NOTE: This method exists only so it can be overriden and disabled in
		# NullZvitWriter.
		#
		if not os.path.isdir (self.logDir):
			raise OSError("The logDir \"{}\" does not exist!".format(self.logDir))
		if os.path.isfile(self.logFilePath):
			raise OSError("The logfile \"{}\" already exists! "
			              "This should not have happened!\n"
			              "Is UUID's entropy source broken?".format(self.logFilePath))
		
		return self
	
	def _commonTagLogic     (self, defaultPluginName, **kwargs):
		"""
		Handle several flags common to most summary log*() methods and
		return a tuple metadata, reject, tag.
		"""
		
		metadata, reject, tag = None, True, None
		
		"""
		Currently we handle the following several pieces of information here:
		"""
		tag           = kwargs.pop("tag")
		collections   = kwargs.pop("collections",   [TfGraphKeys.SUMMARIES])
		tagPrefix     = kwargs.pop("tagPrefix",     None)
		displayName   = kwargs.pop("displayName",   None)
		description   = kwargs.pop("description",   None)
		pluginName    = kwargs.pop("pluginName",    defaultPluginName)
		content       = kwargs.pop("content",       None)
		
		"""
		If the tag prefix is None or True, it is automatically computed from
		the tag scopes. Otherwise, the given tag scope is used.
		"""
		tag = self.getFullyQualifiedTag(tag, tagPrefix)
		
		"""
		Construct metadata object if needed.
		"""
		metadata = convert_metadata(displayName,
		                            description,
		                            pluginName,
		                            content)
		
		"""
		If this tag does not belong to a selected collection, recommend
		rejection of the summary.
		"""
		if self._collectionMatcher is not None:
			for collection in collections:
				if self._collectionMatcher(collection):
					break
			else:
				return metadata, reject, tag
		
		"""
		Check the full tag for a match. Recommend rejecting the summary if
		there is no match.
		"""
		if self._tagMatcher is not None and not self._tagMatcher(tag):
			return metadata, reject, tag
		
		
		"""
		Accept the tag.
		"""
		return metadata, False, tag    # reject=False
	
	def _spawnFlushThread   (self):
		"""
		Spawn the flusher thread, if it hasn't been spawned already, and if we
		have asked for asynchronous writing.
		"""
		
		with self._lock:
			if self.asynchronous and not self._flushThread:
				def flusher():
					"""
					Flusher thread implementation. Simply waits on the
					condition and periodically flush to disk the buffer
					contents.
					
					*MUST NOT* call close(), because it *WILL* lead to deadlock.
					
					On exit request, will indicate its own exit by destroying
					the reference to itself in the ZvitWriter, and will then
					notify all waiters before terminating.
					"""
					
					thrd = threading.currentThread()
					with self._lock:
						while not thrd.isExiting:
							self._lock.wait(self._flushSecs)
							self.flush()
						self._flushThread = None
						self._lock.notifyAll()
				
				
				thrdName = "zvitwriter-{:x}-flushThread".format(id(self))
				self._flushThread = threading.Thread(target=flusher, name=thrdName)
				self._flushThread.isExiting = False
				self._flushThread.start()
		
		return self
	
	def _commitValues       (self):
		"""
		Withdraw every value from the dictionary, construct a summary message
		from them and enqueue it for writeout, but do not flush the bytebuffer.
		"""
		
		with self._lock:
			if self._VB:
				self.writeSummary(TfSummary.fromDict(self._VB))
				self._VB = {}
		return self
	
	def _stageValue         (self, v):
		"""
		Stage a value in the VB.
		"""
		
		assert isinstance(v, TfValue)
		
		with self._lock:
			if hasattr(v, "metadata") and v.tag in self._MB:
				# TensorBoard only keeps the first metadata it sees to save space
				del v.metadata
			if v.tag in self._VB:
				# There is a value with the same tag already staged in the
				# summary value buffer. Commit it to the bytebuffer.
				self._commitValues()
			self._VB[v.tag] = v
			self._MB.add(v.tag)
		return self
	
	#
	# Public API
	#
	def push                (self):
		"""
		Push ourselves onto the thread-local stack of writers.
		"""
		
		l = ZvitWriter._tls.__dict__.setdefault("writerStack", [])
		l.append(self)
		return self
	
	def pop                 (self):
		"""
		Pop ourselves off the thread-local stack of writers.
		"""
		
		return ZvitWriter._tls.writerStack.pop()
	
	def pushTag             (self, groupName):
		assert isinstance(groupName, str)
		assert groupName != ""
		assert "/" not in groupName
		self._tagScopeStack.append(groupName)
		return self
	
	def popTag              (self):
		self._tagScopeStack.pop()
		return self
	
	def flush               (self):
		"""
		Write out and flush the bytebuffer to disk synchronously.
		"""
		
		with self._lock:
			self._commitValues()
			if self._BB:
				with open(self.logFilePath, "ab") as f:
					f.write(self._BB)
					f.flush()
				self._BB = bytearray()
		return self
	
	def close               (self):
		"""
		Close the ZvitWriter.
		
		Specifically, request the flusher thread to exit, then wait for it to
		do so cleanly, and flush anything left in the buffer synchronously.
		"""
		
		with self._lock:
			while self._flushThread:
				self._flushThread.isExiting = True
				self._lock.notifyAll()
				self._lock.wait()
			return self.flush()
	
	def step                (self, step=None):
		"""
		Increment (or set) the global step number.
		"""
		
		with self._lock:
			if step is None:
				"""
				Since the step number is being changed, commit all of the
				waiting summaries to the bytebuffer, so that they are
				recorded with the correct step number.
				"""
				self._commitValues()
				self._globalStep += 1
			else:
				"""
				We're forcibly changing the global step number. We should
				not just commit the waiting summaries to the buffer, but also
				flush it out afterwards.
				"""
				self._commitValues()
				self.flush()
				self._globalStep = int(step)
		return self
	
	def write               (self, b, flush=None):
		"""
		Write raw data to the bytebuffer.
		
		**All** additions to the bytebuffer **must** happen through this
		method. This method and the bytebuffer are the portal between the
		log-generating and log-writing parts of the writer object.
		
		The only time the bytebuffer's size can change other than through this
		method is when it is flushed and emptied periodically from within the
		flush() method, which may be called either synchronously by any thread,
		or asynchronously by the flusher thread.
		
		If the keyword argument flush=True,  flush synchronously the BB.
		If the keyword argument flush=None (the default), flush the BB
		synchronously or asynchronously depending on the flushSecs/flushBufSz
		properties.
		If the keyword argument flush=False, do not flush the BB at all.
		"""
		
		b = bytearray(b)
		l = len(b)
		if l:
			with self._lock:
				#
				# For reasons also elaborated upon at length in __init__(), it
				# is desirable to be as lazy as possible in creating the logfile
				# itself, and thus polluting the filesystem. In the event of this
				# ZvitWriter being created and destroyed without having recorded
				# any data, or the program crashing before being able to do so,
				# we would like to avoid creating the logfile entirely.
				#
				# But even a logfile devoid of any event must have a specific
				# file header (FH), and any data in the bytebuffer (BB) will be
				# written out on flush(), which is called from __del__(). This
				# is problematic for us because it would mean that the logfile
				# would get created on object destruction even if it contains
				# no useful events whatsoever.
				#
				# We work around this by *withholding* the FH from the BB until
				# last possible moment: When a non-zero number of bytes is
				# first written into the BB. At that moment, we prepend the FH
				# into the BB, and the FH, along with a first packet of useful
				# data, become eligible for synchronous or asynchronous flushing
				# to the logfile.
				#
				if self._FH:
					self._BB, self._FH = self._FH, None
				
				# True append into bytebuffer
				self._BB += b
				
				#
				# Flushing policies:
				# - False:           Disabled
				# - None (default):  Synchronously if overflowing;
				#                    Asynchronously otherwise.
				# - True:            Synchronously.
				#
				if   flush is False: pass
				elif flush is None:
					if self.overflowing: self.flush()
					else:                self._spawnFlushThread()
				elif flush is True:  self.flush()
		
		return l
	
	def writeEvent          (self, e):
		"""
		Append a TfEvent to the bytebuffer.
		"""
		
		with self._lock:
			self.write(e.asRecordByteArray())
		return self
	
	def writeSummary        (self, s):
		"""
		Write a TfSummary object to the bytebuffer.
		"""
		
		with self._lock:
			self.writeEvent(s.asEvent(self.globalStep))
		return self
	
	def getFullyQualifiedTag(self, tag, tagPrefix=None):
		"""
		Compute the fully-qualified tag given the partial tag provided, taking
		into account the tag scopes defined so far in the current thread.
		"""
		
		assert not tag.startswith("/") and not tag.endswith("/")
		assert tagPrefix is None or isinstance(tagPrefix, str)
		
		if tagPrefix is None:
			tagPath = self._tagScopeStack + [tag]
		else:
			assert not tagPrefix.startswith("/")
			if tagPrefix == "":
				tagPath = [tag]
			else:
				tagPath = [tagPrefix, tag]
		
		return re.sub("/+", "/", "/".join(tagPath))
	
	@classmethod
	def topZvit             (kls):
		"""
		Return the default ZvitWriter for the current thread.
		
		This will return the ZvitWriter currently top-of-stack on the current
		thread.
		"""
		
		l = ZvitWriter._tls.__dict__.setdefault("writerStack", [])
		return l[-1] if l else NullZvitWriter()
	
	
	#
	# Public API, Logging Methods.
	#
	def logScalar           (self, tag, scalar, **kwargs):
		"""Log a single scalar value."""
		
		metadata, reject, tag = self._commonTagLogic("scalars", tag=tag, **kwargs)
		if reject: return self
		
		val = TfValue(tag=tag, metadata=metadata, simpleValue=scalar)
		return self._stageValue(val)
	
	def logScalars          (self, scalarsDict, **kwargs):
		"""Log multiple scalar values, provided as a (tag, value) iterable."""
		
		with self._lock:
			for tag, scalar in scalarsDict.items():
				self.logScalar(tag, scalar, **kwargs)
		return self
	
	def logImage            (self, tag, images, csc=None, h=None, w=None, maxOutputs=3, **kwargs):
		"""
		Log image(s).
		
		Accepts either a single encoded image as a bytes/bytearray, or one or
		more images as a 3- or 4-D numpy array in the form CHW or NCHW.
		"""
		
		if   isinstance(images, (bytes, bytearray)):
			"""
			"Raw" calling convention: `image` contains an image file, and all
			arguments are mandatory. Image is logged encoded as-is
			"""
			
			metadata, reject, tag = self._commonTagLogic("images", tag=tag+"/image", **kwargs)
			if reject: return self
			
			val = TfImage(height     = int(h),
			              width      = int(w),
			              colorspace = int(csc),
			              imageData  = images).asValue(tag, metadata)
			return self._stageValue(val)
		elif isinstance(images, (list, np.ndarray)):
			"""
			"Numpy" calling convention: `image` is a numpy ndarray shaped (N,C,H,W).
			Conversion is to PNG -z 9. The precise transformation depends on the
			number of channels, datatype and content.
			"""
			
			#
			# Expand dimensionality
			#
			if isinstance(images, np.ndarray) and images.ndim == 3:
				images = images[np.newaxis, ...]
			
			#
			# Iterate.
			#
			for i, image in enumerate(images):
				#
				# Do not output more than the limit of images.
				#
				if i >= maxOutputs:
					break
				
				#
				# Follow TF naming algorithm for image batches.
				#
				if i == 0 and maxOutputs == 1:
					metadata, reject, tag = self._commonTagLogic("images", tag=tag+"/image",         **kwargs)
				else:
					metadata, reject, tag = self._commonTagLogic("images", tag=tag+"/image/"+str(i), **kwargs)
				if reject: continue
				
				#
				# Follow TF type-conversion algorithm for individual images.
				#
				# If   c == 1: Assume grayscale.
				# Elif c == 2: Assume grayscale+alpha.
				# Elif c == 3: Assume RGB.
				# Elif c == 4: Assume RGBA.
				# Else: raise
				#
				c, h, w = image.shape
				if   c == 1:
					csc  = TfColorSpace.GRAYSCALE
					mode = "L"
				elif c == 2:
					csc = TfColorSpace.GRAYSCALE_ALPHA
					mode = "LA"
				elif c == 3:
					csc = TfColorSpace.RGB
					mode = "RGB"
				elif c == 4:
					csc = TfColorSpace.RGBA
					mode = "RGBA"
				else:
					raise ValueError("Invalid image specification!")
				
				#
				# (continued TF type-conversion algorithm for individual images)
				#
				# If   image.dtype == np.uint8:
				#     pass
				# Elif image.min() >= 0:
				#     image /= image.max()/255.0
				#     image  = image.astype(np.uint8)
				# Else:
				#     image.scale( s.t. min >= -127 and max <= 128 )
				#     image += 127
				#
				if   image.dtype == np.uint8:
					pass
				elif image.min() >= 0:
					image *= +255.0/image.max()
				else:
					fMin, fMax = abs(-127.0/image.min()), abs(+128.0/image.max())
					image *= np.minimum(fMin, fMax)
					image += +127.0
				image = image.astype(np.uint8)
				
				#
				# Encode as PNG using an in-memory buffer as the "file" stream.
				#
				
				from PIL.Image import frombytes
				stream = BytesIO()
				image  = frombytes(mode, (w,h), image.transpose(1,2,0).tobytes("C"))
				image.save(stream, format="png", optimize=True)  # Always PNG -z 9
				image = stream.getvalue()
				stream.close()
				
				#
				# Log the image.
				#
				val = TfImage(height     = int(h),
				              width      = int(w),
				              colorspace = int(csc),
				              imageData  = image).asValue(tag, metadata)
				self._stageValue(val)
		else:
			raise ValueError("Unable to interpret image arguments!")
		
		return self
	
	def logAudio            (self, tag, audios, sampleRate, maxOutputs=3, **kwargs):
		"""
		Log audio sample(s).
		
		Accepts either a single or a batch of audio samples as a Numpy 1-D, 2-D
		or 3-D array of floating-point numbers in the range [-1, +1], and
		encodes it to WAVE format.
		
		A 1-D array is assumed to be shaped (Time).
		A 2-D array is assumed to be shaped (Chann,Time).
		A 3-D array is assumed to be shaped (Batch,Chann,Time).
		"""
		
		#
		# Expand dimensionality
		#
		if   isinstance(audios, np.ndarray) and audios.ndim == 1:
			audios = audios[np.newaxis, np.newaxis, ...]
		elif isinstance(audios, np.ndarray) and audios.ndim == 2:
			audios = audios[np.newaxis,             ...]
		
		#
		# Iterate.
		#
		for i, audio in enumerate(audios):
			#
			# Do not output more than the limit of audios.
			#
			if i >= maxOutputs:
				break
			
			#
			# Follow TF naming algorithm for audio batches.
			#
			if i == 0 and maxOutputs == 1:
				metadata, reject, tag = self._commonTagLogic("audio", tag=tag+"/audio",         **kwargs)
			else:
				metadata, reject, tag = self._commonTagLogic("audio", tag=tag+"/audio/"+str(i), **kwargs)
			if reject: continue
			
			#
			# If audios is a list, we must ensure the presence of a channels axis.
			# Then, in WAV, audio frames are interleaved, so we must transpose to (T,C).
			# Lastly, we want to encode as 16-bit signed integer:
			#
			
			if audio.ndim == 1:
				audio = audio[np.newaxis, ...]
			audio  = audio.transpose()
			audio *= 32767.0
			audio  = audio.astype(np.int16)
			lengthFrames = audio.shape[0]
			numChannels  = audio.shape[1]
			
			#
			# Always encode the audio as 16-bit integer WAVE.
			#
			import wave
			stream = BytesIO()
			wavewr = wave.open(stream, "wb")
			wavewr.setnchannels(numChannels)
			wavewr.setframerate(sampleRate)
			wavewr.setsampwidth(2) # 16-bit integer
			wavewr.writeframes(audio.tobytes("C"))
			wavewr.close()
			audio = stream.getvalue()
			stream.close()
			
			#
			# Log the audio.
			#
			val   = TfAudio(sampleRate   = sampleRate,
			                numChannels  = numChannels,
			                lengthFrames = lengthFrames,
			                audioData    = audio,
			                contentType  = "audio/wav").asValue(tag, metadata)
			self._stageValue(val)
		
		return self
	
	def logText             (self, tag, text,   dimNames=None, **kwargs):
		"""
		Log a tensor of text strings to the "Text" dashboard.
		"""
		
		metadata, reject, tag = self._commonTagLogic("text", tag=tag, **kwargs)
		if reject: return self
		
		val = TfTensor.fromText (text,   dimNames).asValue(tag, metadata)
		return self._stageValue(val)
	
	def logTensor           (self, tag, tensor, dimNames=None, **kwargs):
		"""
		Log a tensor.
		"""
		
		metadata, reject, tag = self._commonTagLogic(None, tag=tag, **kwargs)
		if reject: return self
		
		val = TfTensor.fromNumpy(tensor, dimNames).asValue(tag, metadata)
		return self._stageValue(val)
	
	def logHist             (self, tag, hist,   bucketLimits=None, **kwargs):
		"""
		Log a histogram.
		"""
		
		metadata, reject, tag = self._commonTagLogic("histograms", tag=tag, **kwargs)
		if reject: return self
		
		if isinstance(hist, HistogramAccumulator):
			hist = TfHistogram(**hacc)
		else:
			hist = TfHistogram.fromNumpy(hist, bucketLimits=bucketLimits)
		
		val = hist.asValue(tag, metadata)
		return self._stageValue(val)
	
	def logMessage          (self, msg, level=TfLogLevel.UNKNOWN):
		"""
		Log a message.
		"""
		
		with self._lock:
			#
			# As a special case, log messages always provoke the commit of
			# all accumulated summaries and their synchronous flushing to disk
			# immediately afterwards in order to ensure that
			#
			#   1) Any summaries about which the message might be are temporally
			#      ordered *before* the log message, consistent with the order
			#      they were generated in.
			#   2) The log message and summaries are made immediately visible
			#      on-disk, to allow for debugging in case of a crash soon
			#      afterwards. Otherwise, the messages might be lost along with
			#      the rest of the in-memory bytebuffer.
			#
			
			self._commitValues()
			self.writeEvent(TfLogMessage(level   = level,
			                             message = msg).asEvent(self.globalStep))
			self.flush()
		return self
	
	def logSession          (self, status, msg=None, path=None):
		"""
		Log a session status change.
		"""
		
		with self._lock:
			#
			# As a special case, session log messages always provoke the
			# commit of all accumulated summaries and their synchronous
			# flushing to disk immediately afterwards in order to ensure that:
			#
			#   1) All summaries recorded before a session status change are
			#      temporally ordered *before* it, consistent with the order
			#      they were generated in.
			#   2) Session log messages are sufficiently rare and important
			#      that they deserve immediate writeout.
			#
			
			self._commitValues()
			self.writeEvent(TfSessionLog(status         = status,
			                             message        = msg,
			                             checkpointPath = path).asEvent(self.globalStep))
			self.flush()
		return self
	
	def logLayout           (self, layout):
		"""
		Log a custom scalars chart layout.
		
		layout must be a TfLayout object.
		"""
		
		#
		# The serialized form of a layout is a TfTensor of datatype
		# TfDataType.STRING, whose string payload is the encoded TfLayout
		# protobuf message. The tensor is logged as a TfValue with the magic
		# tag "custom_scalars__config__" and the pluginName "custom_scalars".
		#
		assert isinstance(layout, TfLayout)
		metadata = convert_metadata(pluginName="custom_scalars")
		val      = TfTensor.fromText(layout.asByteArray())                  \
		                   .asValue("custom_scalars__config__", metadata)
		return self._stageValue(val)
	
	#
	# Per-class, thread-local data
	#
	_tls             = threading.local()
	
	#
	# snake_case aliases for those who prefer that.
	#
	top_zvit         = topZvit
	log_scalar       = logScalar
	log_scalars      = logScalars
	log_image        = logImage
	log_audio        = logAudio
	log_text         = logText
	log_tensor       = logTensor
	log_hist         = logHist
	log_message      = logMessage
	log_session      = logSession
	log_layout       = logLayout


#
# Null Zvit Writer
#

class NullZvitWriter(ZvitWriter):
	"""
	Null ZvitWriter.
	
	Used when one needs a "null", "sink" or "/dev/null" ZvitWriter instance.
	
	It's identical in interface to a ZvitWriter, but it has been defanged.
	Null ZvitWriters will still support tag scopes and such, but will not log
	anything whatsoever to any file, nor spawn any logging thread to do so.
	"""
	
	def __init__            (self, *args, **kwargs):
		if "logDir" in kwargs:  kwargs["logDir"] = ""
		elif len(args) > 0:     args             = ("",)+args[1:]
		else:                   args             = [""]
		super().__init__(*args, **kwargs)
	def _initAsserts        (self):                return self
	def _spawnFlushThread   (self):                return self
	def write               (self, b, flush=None): return len(b)
	def flush               (self):                return self
	def close               (self):                return self




#
# Global convenience functions exposing the writer API, using the
# thread-local top-of-stack ZvitWriter.
#

@contextlib.contextmanager
def tagscope        (*args, **kwargs):
	with topZvit().tagscope(*args, **kwargs) as z:
		yield z
def topZvit         ():
	return ZvitWriter.topZvit()
def logScalar       (*args, **kwargs):
	return topZvit().logScalar (*args, **kwargs)
def logScalars      (*args, **kwargs):
	return topZvit().logScalars(*args, **kwargs)
def logImage        (*args, **kwargs):
	return topZvit().logImage  (*args, **kwargs)
def logAudio        (*args, **kwargs):
	return topZvit().logAudio  (*args, **kwargs)
def logText         (*args, **kwargs):
	return topZvit().logText   (*args, **kwargs)
def logTensor       (*args, **kwargs):
	return topZvit().logTensor (*args, **kwargs)
def logHist         (*args, **kwargs):
	return topZvit().logHist   (*args, **kwargs)
def logMessage      (*args, **kwargs):
	return topZvit().logMessage(*args, **kwargs)
def logSession      (*args, **kwargs):
	return topZvit().logSession(*args, **kwargs)
def logLayout       (*args, **kwargs):
	return topZvit().logLayout (*args, **kwargs)
top_zvit    = topZvit
log_scalar  = logScalar
log_scalars = logScalars
log_image   = logImage
log_audio   = logAudio
log_text    = logText
log_tensor  = logTensor
log_hist    = logHist
log_message = logMessage
log_session = logSession
log_layout  = logLayout

