# -*- coding: utf-8 -*-

#
# Imports
#

import numpy                         as np
import os
from   zvit                      import *



#
# Mandelbrot generation
#

def mandelbrotLattice():
	x      = np.linspace(-2.4, +1.0, 1700)
	y      = np.linspace(+1.2, -1.2, 1200)
	r, c   = np.meshgrid(x, y)
	return   r*1.0 + c*1.0j

def mandelbrotStep(z, g):
	z = np.where(np.absolute(z) > 1e4, z, z*z+g)
	
	#
	# Logging can be done in nested scopes. The global methods log*() retrieve
	# a default logger from a stack (which is per-thread), and each logger
	# also manages a stack of tag scopes (per-logger, per-thread)
	#
	# Creating a tag scope causes all log*() functions and methods to prefix
	# the tag with the name of the scopes, separated with forward slashes (/).
	#
	# In the example below, two histogram metrics are collected with the tags
	#
	#     `/histo/grams/zMag` and
	#     `/histo/grams/zAng`
	#
	
	with tagscope("histo", "grams"):
		logHist("zMag", np.log(np.absolute(z)+1e-8),
		        bucketLimits = np.linspace(-5, +12, 10000),
		        displayName  = "Log Z Magnitude",
		        description  = "Histogram of logarithms of complex number magnitudes.")
	
	with tagscope("histo"):
		with tagscope("grams"):
			logHist("zAng", np.angle(z),
			        bucketLimits = np.linspace(-np.pi, +np.pi, 1000),
			        displayName  = "Z Angle",
			        description  = "Histogram of complex number angles.")
	
	return z

def mandelbrotEscTime(escTime, z, i):
	#
	# Boolean escape conditions.
	#
	# A point in a Mandelbrot set is deemed to have "escaped" if it reaches
	# a magnitude of >2, since it is then guaranteed to be ejected to
	# infinity. The escape time (in # of steps) can be used to plot a beautiful
	# visualization of the Mandelbrot set by making it correspond to colors
	# in a palette.
	#
	
	zMag             = np.absolute(z)
	currentlyEscaped = zMag    > 2
	alreadyEscaped   = escTime < i
	newlyEscaped     = currentlyEscaped & ~alreadyEscaped
	
	#
	# One can also put the hierarchy directly in the tag name.
	#
	logScalar("nested/scopes/escaped", np.mean(currentlyEscaped))
	
	
	# Smooth escape-time estimate
	antiNaN          = 4.0*~newlyEscaped
	smoothTime       = i+1 - np.log(0.5 * np.log(zMag+antiNaN)) / np.log(2)
	
	#
	# Masked computation of escape time:
	#
	# If   already escaped:
	#     pass
	# Elif newly escaped:
	#     set i+1 - log(0.5*log(|z|))/log(2)
	# Else
	#     set i+1
	#
	
	escTime = np.where(currentlyEscaped, escTime,    i+1)
	escTime = np.where(newlyEscaped,     smoothTime, escTime)
	
	# Vizualization
	d2      = (i+1)/2.0
	img     = escTime/(i+1)
	palette = escTime > d2 # If False, Black-to-Red; If True, Red-to-White.
	imgR    = np.where(palette, 1,                   escTime/d2)
	imgGB   = np.where(palette, (escTime-d2)/(d2+1), 0)
	imgR    = np.where(currentlyEscaped, imgR,  0)
	imgGB   = np.where(currentlyEscaped, imgGB, 0)
	img     = np.stack([imgR, imgGB, imgGB], axis=0)
	
	logImage("viz", img).close()
	
	##DEBUG: OpenCV viewing code
	#import cv2 as cv
	#cvImg = (img[::-1]*255).transpose(1,2,0).astype(np.uint8)
	#cv.imshow("Image", cvImg)
	#cv.waitKey(30)
	
	# Can put rank-0 tensors into the Scalars viewing pane with pluginName="scalars".
	logTensor("test/rank0tensor", img.astype("float32").sum(), pluginName="scalars")
	
	return escTime

def seedWithString(seed):
	#
	# np.random.seed() won't use anything more than 32 seed bits, and we are
	# providing the seed as a string, not an integer.
	#
	# Be brutal and directly generate an entire MT19937 state using a similar
	# process as specified by WPA2 for Pairwise Master Key derivation:
	# 
	#     PBKDF2(hash_name = SHA1,
	#            password  = seed,
	#            salt      = publicSeed,
	#            rounds    = 4096,
	#            dk_bytes  = numBytes)
	#
	# but with the following provisions:
	# 
	#     - We use SHA1 rather than HMAC-SHA1 because that is what
	#       hashlib supports.
	#     - The password is the seed, as a UTF-8 string.
	#     - The salt is the user's public seed as retrieved from
	#       the environment variable PUBLIC_SEED, as a UTF-8 string.
	#       If it is not specified, use
	#           "5a8279996ed9eba18f1bbcdcca62c1d6"
	#       the concatenation of SHA1's round constants, which are
	#       themselves the square roots of 2,3,5,10 multiplied by 2**30.
	#     - The number of bytes of derived key material is 624*4 bytes
	#       rather than 256 bits because Numpy's MT19937's state is
	#       624 words large.
	#
	publicSeed = os.environ.get("PUBLIC_SEED", "5a8279996ed9eba18f1bbcdcca62c1d6")
	state = np.random.get_state()
	state = (
	    state[0],
	    np.frombuffer(hashlib.pbkdf2_hmac(
	                      "sha1",                     # Hash
	                      seed.encode("utf-8"),       # Password
	                      publicSeed.encode("utf-8"), # Salt
	                      4096,                       # Rounds
	                      state[1].nbytes),           # DKLen
	                  dtype=np.uint32),
	    624,
	    0,
	    0.0,
	)
	np.random.set_state(state)


#
# Run Mandelbrot for 50 steps.
#

def main(a):
	#
	# Create log directory.
	#
	if not os.path.isdir(a.logDir):
		os.mkdir(a.logDir)
	
	#
	# Run Mandelbrot set experiment. Will log to tfevents.*.zvit file every
	# 60 seconds.
	#
	# If resuming from a snapshot, make SURE that `step` is set to the step #
	# at the last snapshot. ALL data with `step` >= than the one provided
	# here WILL BE IGNORED by TensorBoard ("orphaned").
	#
	with ZvitWriter(a.logDir, 0, flushSecs=60.0) as z:
		#
		# Hello World of ZvitWriter: Text
		#
		logText("welcome", [["Hello", ",", "World", "!"],
		                    ["This", "is", "ZvitWriter", "!"]])
		
		#
		# Another random thing: (Stereo) Audio
		#
		lChann = np.sin(2*np.pi*1500*np.arange(16000)/8000)
		rChann = np.sin(2*np.pi* 750*np.arange(16000)/8000)
		logAudio("sounds/twosines",
		         np.stack([lChann, rChann], axis=0),
		         8000,
		         displayName="Two Pure Sines",
		         description="Left: 1500Hz  Right: 750 Hz")
		
		#
		# More random things: Custom Scalars
		#
		logLayout(TfLayout(category=[
			TfCategory(title="Category Title", closed=False, chart=[
				TfChart(title="Sine vs Cosine", multiline=TfMultilineChart(tag=["trig/.*sine"])),
			])
		]))
		
		#
		# Initialize Mandelbrot iteration scheme
		#
		g       = mandelbrotLattice()
		x       = g.copy()
		escTime = np.zeros_like(x, dtype="float64")
		
		#
		# Run the Mandelbrot iterated function for 50 steps, recording the
		# "escape time" of every point.
		#
		for i in range(50):
			x       = mandelbrotStep   (x,          g)
			escTime = mandelbrotEscTime(escTime, x, i)
			
			#
			# The current top-of-stack (default) ZvitWriter can be retrieved with
			# the global function topZivt() or the class method ZvitWriter.topZvit(),
			# and it has methods that correspond to the global log*() functions.
			#
			topZvit().logMessage("Step {:2d} done!".format(i), TfLogLevel.INFO)
			z        .logMessage("Now we log a random scalar", TfLogLevel.WARN)
			
			#
			# Moreover, these methods support a fluent interface through
			# method call chaining:
			#
			#     logMessage("foo", foo).logScalar("bar", bar).logAudio("baz", baz)
			#
			logScalar("loss", 0.65 * 0.95**i).step().flush()
			
			#
			# For the sake of generating material for the custom scalar plugin...
			#
			with tagscope("trig"):
				logScalar("cosine", np.cos(2*np.pi*i/10))
				logScalar("sine",   np.sin(2*np.pi*i/10))
			
			#
			# Give some hint of progress on the command-line...
			#
			print("Step {:2d}".format(i))
	
	#
	# Exit.
	#
	return 0


