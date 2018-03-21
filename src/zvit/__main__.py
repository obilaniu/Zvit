#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
import nauka, sys


class root(nauka.ap.Subcommand):
	class bench(nauka.ap.Subcommand):
		"""Benchmark Zvit internals."""
		parserArgs = {"help": __doc__}
		
		@classmethod
		def run(cls, a):
			from zvit.pb.crc_native import crc32c_update
			import time
			s = 100 * 1024**2
			b = "\x00" * s
			t =-time.time()
			c = crc32c_update(0, b)
			t+= time.time()
			print("CRC32C: {:.3f} MiB/s".format(s/t/(1024**2)))
			return 0
	
	class demo(nauka.ap.Subcommand):
		"""\
		Demo application for Zvit.
		
		Iteratively generates a Mandelbrot and logs its work to a .zvit file. To
		view its contents, launch TensorBoard, pointing it to the log directory
		containing these .zvit files.
		"""
		parserArgs = {"help": __doc__}
		
		@classmethod
		def addArgs(cls, argp):
			argp.add_argument("-d", "--logDir",         default=".",          type=str,
			    help="Path to log directory.")
			argp.add_argument("-s", "--seed",           default=0x6a09e667f3bcc908, type=int,
			    help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
		
		@classmethod
		def run(cls, a):
			from zvit.bin.demo import main
			return main(a)


def main(argv=sys.argv):
	argp = root.addAllArgs()
	try:    import argcomplete; argcomplete.autocomplete(argp)
	except: pass
	a    = argp.parse_args(argv[1:])
	a.__argv__ = argv
	return a.__cls__.run(a)


if __name__ == "__main__":
	sys.exit(main(sys.argv))
