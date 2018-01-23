# -*- coding: utf-8 -*-

# Imports.
import argparse                             as Ap
import hashlib
import sys



#############################################################################################################
##############################                   Subcommands               ##################################
#############################################################################################################

class Subcommand(object):
	name  = None
	
	@classmethod
	def addArgParser(cls, subp, *args, **kwargs):
		argp = subp.add_parser(cls.name, help=cls.__doc__, *args, **kwargs)
		cls.addArgs(argp)
		argp.set_defaults(__subcmdfn__=cls.run)
		return argp
	
	@classmethod
	def addArgs(cls, argp):
		pass
	
	@classmethod
	def run(cls, a):
		pass


class Bench(Subcommand):
	"""\
	Benchmark Zvit internals.
	"""
	
	name = "bench"
	
	@classmethod
	def run(cls, a):
		import time, zvit.pb.crc_native
		s = 100 * 1024**2
		b = "\x00" * s
		t = -time.time()
		c = zvit.pb.crc_native.crc32c_update(0, b)
		t+= time.time()
		
		print("CRC32C: {:.3f} MiB/s".format(s/t/(1024**2)))
		
		return 0


class Demo(Subcommand):
	"""\
	Demo application for Zvit.
	
	Iteratively generates a Mandelbrot and logs its work to a .zvit file. To
	view its contents, launch TensorBoard, pointing it to the log directory.
	"""
	
	name = "demo"
	
	@classmethod
	def addArgs(cls, argp):
		argp.add_argument("-d", "--logDir",
		    default=".",
		    type=str,
		    help="Path to log directory.")
		argp.add_argument("-s", "--seed",
		    default="0",
		    type=str,
		    help="Seed for PRNGs, as a string. Default is \"0\".")
	
	@classmethod
	def run(cls, a):
		from .demo import main
		return main(a)




#############################################################################################################
##############################               Argument Parsers               #################################
#############################################################################################################

def getArgParser(prog):
	argp = Ap.ArgumentParser(prog        = prog,
	                         usage       = None,
	                         description = None,
	                         epilog      = None)
	subp = argp.add_subparsers()
	argp.set_defaults(argp=argp)
	argp.set_defaults(subp=subp)
	
	# Add global args to argp here?
	# ...
	
	# Add subcommands
	for k, v in globals().items():
		if(isinstance(v, type)       and
		   issubclass(v, Subcommand) and
		   v != Subcommand):
			v.addArgParser(subp)
	
	# Return argument parser.
	return argp


#
# Main
#
def main(a):
	a = getArgParser(a[0]).parse_args(a[1:])
	return a.__subcmdfn__(a)
