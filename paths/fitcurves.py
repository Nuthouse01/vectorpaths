"""Code to fit Cubic Beziers to a set of points.

The original code was C++ code by Philip J. Schneider and published in
'Graphics Gems' (Academic Press, 1990) 'Algorithm for Automatically Fitting
Digitized Curves'.  This code is based on a Python implementation by
Volker Poplawski (Copyright (c) 2014).
"""
import numpy as np
import logging

from . import CubicBezier
from . import _path_logger


def fit_cubic_bezier(xc, yc, max_error):
	"""Fit a set of cubic bezier curves to points (xc,yc).

	:param xc: (n,) array of x coordinates of the points to fit.
	:param yc: (n,) array of y coordinates of the points to fit.
	:param max_error: maximum error to accept (in units of xc and yc).
	"""
	_path_logger.debug('Fitting points')

	if len(xc)!=len(yc):
		raise ValueError('Number of x and y points does not match')
	p = np.zeros((len(xc),2))
	p[:,0] = xc
	p[:,1] = yc

	# Compute unit tangents at the end points of the points.
	left_tangent = _normalise(p[1,:]-p[0,:])
	right_tangent = _normalise(p[-2,:]-p[-1,:])

	return _fit_cubic(p, left_tangent, right_tangent, max_error, 0)

def _fit_cubic(p, left_tangent, right_tangent, max_error, depth, max_reparam_iter=20):
	"""Recursive routine to fit cubic bezier to a set of data points."""

	# Use heuristic if region only has two points in it.
	if (len(p) == 2):
		_path_logger.debug('Recursion depth {}: Only two points to fit, using heuristic'.format(depth))
		dist = np.linalg.norm(p[0,:] - p[1,:])/3.0
		left = left_tangent*dist
		right = right_tangent*dist
		return [CubicBezier([p[0,:], p[0,:]+left, p[1,:]+right, p[1,:]])]

	# Parameterise points and try to fit curve.
	u = _chord_length_parameterise(p)
	bezier = generate_bezier(p, u, left_tangent, right_tangent)

	# Find max deviation of points to fitted curve; if the maximum error is
	# less than the error then return this bezier.
	error, split_point = _compute_max_error(p, bezier, u)
	if error<max_error:
		_path_logger.debug('Recursion depth {}: Optimal solution found'.format(depth))
		return [bezier]

	# If error not too large, try some reparameterization and iteration.
	if error < max_error**2:
		for i in range(max_reparam_iter):
			_path_logger.debug('Recursion depth {}: Reparameterising step {:2d}/{:2d}'.format(depth, i, max_reparam_iter))
			uprime = _reparameterise(bezier, p, u)
			bezier = generate_bezier(p, uprime, left_tangent, right_tangent)
			error, split_point = _compute_max_error(p, bezier, uprime)
			if error<max_error:
				_path_logger.debug('Recursion depth {}: Optimal reparameterised solution found'.format(depth))
				return [bezier]
			u = uprime
		_path_logger.debug('Recursion depth {}: No optimal reparameterised solution found with error {} and split={} and length={}'.format(depth, error, split_point, len(p)))

	# We can't refine this anymore, so try splitting at the maximum error point
	# and fit recursively.
	_path_logger.debug('Recursion depth {}: Splitting'.format(depth))
	beziers = []
	centre_tangent = _normalise(p[split_point-1,:] - p[split_point+1,:])
	beziers += _fit_cubic(p[:split_point+1,:], left_tangent, centre_tangent, max_error, depth+1)
	beziers += _fit_cubic(p[split_point:,:], -centre_tangent, right_tangent, max_error, depth+1)

	return beziers


def generate_bezier(p, u, left_tangent, right_tangent):
	bezier = CubicBezier([p[0,:], [0,0], [0,0], p[-1,:]])

	# Compute the A matrix.
	A = np.zeros((len(u), 2, 2))
	A[:,0,:] = left_tangent[None,:] * (3*((1-u[:,None])**2))*u[:,None]
	A[:,1,:] = right_tangent[None,:] * 3*(1-u[:,None])*u[:,None]**2

	# Compute the C and X matrixes
	C = np.zeros((2, 2))
	X = np.zeros(2)
	for i in range(len(u)):
		C[0,0] += np.dot(A[i,0,:], A[i,0,:])
		C[0,1] += np.dot(A[i,0,:], A[i,1,:])
		C[1,0] += np.dot(A[i,1,:], A[i,0,:])
		C[1,1] += np.dot(A[i,1,:], A[i,1,:])

		tmp = [p[i,0] - CubicBezier._q([p[0,0], p[0,0], p[-1,0], p[-1,0]], u[i]),
				p[i,1] - CubicBezier._q([p[0,1], p[0,1], p[-1,1], p[-1,1]], u[i])]

		X[0] += np.dot(A[i][0], tmp)
		X[1] += np.dot(A[i][1], tmp)

	# Compute the determinants of C and X.
	det_C0_C1 = C[0,0]*C[1,1] - C[1,0]*C[0,1]
	det_C0_X  = C[0,0]*X[1] - C[1][0]*X[0]
	det_X_C1  = X[0]*C[1,1] - X[1]*C[0,1]

	# Finally, derive alpha values
	alpha_l = 0.0
	if det_C0_C1!=0:
		alpha_l = det_X_C1/det_C0_C1
	alpha_r = 0.0
	if det_C0_C1!=0:
		alpha_r = det_C0_X/det_C0_C1

	# If either alpha negative then we use the Wu/Barsky heuristic, and if
	# alpha is zero then there are coincident control points that give a
	# divide by zero during Newton-Raphson iteration.
	seg_length = np.linalg.norm(p[0,:] - p[-1,:])
	epsilon = 1.0e-6 * seg_length
	if alpha_l<epsilon or alpha_r<epsilon:
		# fall back on standard (probably inaccurate) formula, and subdivide further if needed.
		bezier.px[1] = bezier.px[0] + left_tangent[0]*seg_length/3.0
		bezier.py[1] = bezier.py[0] + left_tangent[1]*seg_length/3.0
		bezier.px[2] = bezier.px[3] + right_tangent[0]*seg_length/3.0
		bezier.py[2] = bezier.py[3] + right_tangent[1]*seg_length/3.0

	else:
		# First and last control points of the Bezier curve are positioned
		# exactly at the first and last data points.  Control points 1 and 2
		# are positioned an alpha distance on the left and right tangent
		# vectors left.
		bezier.px[1] = bezier.px[0] + left_tangent[0]*alpha_l
		bezier.py[1] = bezier.py[0] + left_tangent[1]*alpha_l
		bezier.px[2] = bezier.px[3] + right_tangent[0]*alpha_r
		bezier.py[2] = bezier.py[3] + right_tangent[1]*alpha_r

	return bezier

def _chord_length_parameterise(p):
	"""Assign parameter values to points using relative distances"""

	rel_dist = np.zeros(len(p))
	rel_dist[1:] = np.linalg.norm(p[1:,:]-p[0:-1,:])
	u = np.cumsum(rel_dist)
	u /= u[-1]

	return u


def _reparameterise(bezier, p, u):
	delta = bezier.xy(u) - p
#	print(delta)
	numerator = np.sum(delta*bezier.xyprime(u))
	denominator = np.sum(bezier.xyprime(u)**2 + delta*bezier.xyprimeprime(u))
	if denominator==0.0:
		return u
	else:
		return u - numerator/denominator


def _compute_max_error(p, bezier, u):
	"""Compute the maximum error between a set of points and a bezier curve"""
	max_dist = 0.0
	split_point = len(p)//2

	dists = np.linalg.norm(bezier.xy(u)-p,axis=1)
	i = np.argmax(dists)

	if i==0:
		return 0.0, len(p)//2
	elif i==len(p)-1:
		return 0.0, len(p)//2
	else:
		return dists[i], i


def _normalise(v):
	"""Normalise a vector"""
	if len(np.array(v).shape)==1:
		return v/np.linalg.norm(v)
	else:
		return np.divide(v, np.linalg.norm(v,axis=1)[:,None])
