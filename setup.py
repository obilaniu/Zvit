#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Imports
#
import os, sys, subprocess, time
from   setuptools import setup, find_packages, Extension


#
# Versioning
#

def getVersionInfo():
	zvitVerMajor  = 0
	zvitVerMinor  = 0
	zvitVerPatch  = 1
	zvitVerIsRel  = True
	zvitVerPreRel = "dev0"
	
	zvitVerShort  = "{zvitVerMajor}.{zvitVerMinor}.{zvitVerPatch}".format(**locals())
	zvitVerNormal = zvitVerShort
	zvitVerGit    = getGitVer()
	zvitEpochTime = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
	zvitISO8601   = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(zvitEpochTime))
	if zvitVerIsRel or not zvitVerGit:
		zvitVerSemVer = zvitVerNormal
		zvitVerFull   = zvitVerNormal
	else:
		zvitVerSemVer = zvitVerNormal+"-"+zvitVerPreRel+"+"+zvitISO8601+".g"+zvitVerGit
		zvitVerFull   = zvitVerNormal+".dev0+"+zvitVerGit
	
	return locals()

def getRoot():
	return os.path.dirname(__file__) or "."

def getGitVer():
	cwd = getRoot()
	if not os.path.isdir(os.path.join(cwd, ".git")):
		return ""
	
	env = os.environ.copy()
	env['LANGUAGE'] = env['LANG'] = env['LC_ALL'] = 'C'
	
	try:
		zvitVerGit = subprocess.Popen(
		    ["git", "rev-parse", "HEAD"],
		    stdout = subprocess.PIPE,
		    stderr = subprocess.PIPE,
		    cwd    = cwd,
		    env    = env
		).communicate()[0].strip().decode("ascii")
	except OSError:
		zvitVerGit = ""
	
	if zvitVerGit == "HEAD":
		zvitVerGit = ""
	
	return zvitVerGit

def writeVersionFile(f, versionInfo):
	f.write("""\
#
# THIS FILE IS GENERATED FROM ZVIT SETUP.PY
#
short_version = "{zvitVerShort}"
version       = "{zvitVerNormal}"
full_version  = "{zvitVerFull}"
git_revision  = "{zvitVerGit}"
sem_revision  = "{zvitVerSemVer}"
release       = {zvitVerIsRel}
build_time    = "{zvitISO8601}"
""".format(**versionInfo))

def getDownloadURL(v):
	return "https://github.com/obilaniu/Zvit/archive/v{}.tar.gz".format(v)




if __name__ == "__main__":
	#
	# Defend against Python2
	#
	if sys.version_info[0] < 3:
		sys.stdout.write("This package is Python 3+ only!\n")
		sys.exit(1)
	
	#
	# Handle version information
	#
	versionInfo = getVersionInfo()
	zvitVerFull = versionInfo["zvitVerFull"]
	with open(os.path.join(getRoot(), "zvit", "version.py"), "w") as f:
		writeVersionFile(f, versionInfo)
	
	
	#
	# Setup
	#
	setup(
	    #
	    # The basics
	    #
	    name                 = "zvit",
	    version              = zvitVerFull,
	    author               = "Olexa Bilaniuk",
	    author_email         = "anonymous@anonymous.com",
	    license              = "MIT",
	    url                  = "https://github.com/obilaniu/Zvit",
	    download_url         = getDownloadURL(zvitVerFull),
	    
	    #
	    # Descriptions
	    #
	    description          = ("A standalone, lightweight logging package "
	                            "that writes TensorFlow tfevents files "
	                            "compatible with TensorBoard."),
	    
	    long_description     =
	    """\
Zvit is a logging package that makes it easy to write TensorFlow Events
files compatible with TensorBoard.

The name "Zvit" is a phonetic transliteration of the Ukrainian word "Звіт",
meaning "report" or "account", typically of a written form.""",
	    
	    classifiers          = [
	        "Development Status :: 1 - Planning",
	        "Environment :: Console",
	        "Intended Audience :: Developers",
	        "Intended Audience :: Science/Research",
	        "License :: OSI Approved :: MIT License",
	        "Operating System :: MacOS",
	        "Operating System :: MacOS :: MacOS X",
	        "Operating System :: POSIX",
	        "Operating System :: Unix",
	        "Programming Language :: Python",
	        "Programming Language :: Python :: 3",
	        "Programming Language :: Python :: 3.4",
	        "Programming Language :: Python :: 3.5",
	        "Programming Language :: Python :: 3.6",
	        "Programming Language :: Python :: 3.7",
	        "Programming Language :: Python :: 3 :: Only",
	        "Topic :: Scientific/Engineering",
	        "Topic :: Scientific/Engineering :: Artificial Intelligence",
	        "Topic :: Scientific/Engineering :: Mathematics",
	        "Topic :: Scientific/Engineering :: Visualization",
	        "Topic :: Software Development",
	        "Topic :: Software Development :: Libraries",
	        "Topic :: Software Development :: Libraries :: Python Modules",
	        "Topic :: System :: Logging",
	        "Topic :: Utilities",
	    ],
	    
	    # Sources
	    packages             = find_packages(exclude=["scripts"]),
	    ext_modules          = [
	        Extension("zvit.pb.crc_native",
	                  [os.path.join(getRoot(), "zvit", "pb", "crc_native.c")])
	    ],
	    scripts              = [
	        "scripts/zvit",
	    ],
	    install_requires     = [
	        "numpy>=1.10",
	        "Pillow>=4.0.0",
	    ],
	    
	    # Misc
	    zip_safe             = False,
	)

