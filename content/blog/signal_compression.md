+++
title = "Signal Compression from a Linear Algebra Perspective"
date = "2025-07-10T21:31:30-04:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

One of my favourite courses from my time at the University of Waterloo was [AMATH 391](https://uwflow.com/course/amath391).
At the time, it had the provocative title *From Fourier to Wavelets*.
Nowadays, it has the more sober name *Data Analysis with Fourier and Wavelet Methods*.
Honestly, this is probably a better name for the course anyways.

I want to present to you a somewhat simple concept from this course.

When we take the Fourier transformation of a signal \(f[t]\), we are finding the infinite dimensional *vector*
that represents the signal in the Hilbert space of (usually) continuous functions (call it \(\mathcal{C}\))
Moreover, in the case of a Fourier analysis, we are representing this *vector* using a "basis" of
infinitely many sinusoids.

Since we are operating in a Hilbert space, each *coordinate* of this *vector* has some magnitude.
What we can then do, is take the \(N\) largest magnitude coordinates from this vector, and set the rest to 0.
Of course, this does result in some loss, but most signals are sparse in the frequency domain anyways.

From a more practical perspective, when you take a digital signal of finite length, the maximum bandwidth of
the signal is at most half of your sampling frequency.
 > For example, a typical audio file will have a sampling frequency of 44.1 kHz, so your audio signal will be able
 > to represent at most 22.05 kHz.

So now imagine a 22,050 degree vector, where each coordinate is some real number.
We can erase all but the 100 largest coordinates, and we will end up with over 90% of the fidelity with a
tiny fraction of the data.

