# basicft8

basicft8.py is a simple demodulator for the WSJT-X FT8 mode. It misses many
possible decodes, but in return it's relatively easy to understand.
You can see a detailed explanation of the code at
http://www.rtmrtm.org/basicft8/ .
Please feel free to copy and modify this code for your own amusement.
I've found that writing my own demodulators for FT8, JT65, etc
has been very interesting and educational.

The code is written in Python, and will work with either Python 3 or
Python 2.7. You'll need a few extra Python packages. Here's how to
install them on Ubuntu Linux:
```
  sudo apt-get install python-numpy
  sudo apt-get install python-pyaudio
```

If you have a Mac with macports:
```
  sudo port install py-numpy
  sudo port install py-pyaudio
```

For Windows:
```
  pip install numpy
  pip install pyaudio
```

This repository contains a few sample FT8 .wav files to help test the
demodulator. To read one, provide the file name on the command line,
like this:

```
  python basicft8.py samples/170923_082000.wav
```

When I do this, I see two lines of output corresponding to two decodes:
```
  1962.5 CQ FK8HA  RG37
  1237.5 ZL1RPL YV5DRN RRR 
```

There are at least seven more signals that could be decoded, but that
would require more sophisticated demodulation algorithms.

To listen to audio from the default sound source (presumably hooked up
to a radio set to e.g. 14.074 USB), try this:

```
  python basicft8.py
```

Robert, AB1HL
