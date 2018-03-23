import numpy as np
import os, sys

def calc_iou(box1, box2):
	# box: [xmin, ymin, xmax, ymax]
	iou = 0.0
	if box1[2] <= box1[0] or box1[3] <= box1[1]:
		return iou
	if box2[2] <= box2[0] or box2[3] <= box2[1]:
		return iou		
	if box1[2] <= box2[0] or box1[0] >= box2[2]:
		return iou
	if box1[3] <= box2[1] or box1[1] >= box2[3]:
		return iou

	xl_min = min(box1[0], box2[0])
	xl_max = max(box1[0], box2[0])
	xr_min = min(box1[2], box2[2])
	xr_max = max(box1[2], box2[2])

	yl_min = min(box1[1], box2[1])
	yl_max = max(box1[1], box2[1])
	yr_min = min(box1[3], box2[3])
	yr_max = max(box1[3], box2[3])

	inter = float(xr_min-xl_max)*float(yr_min-yl_max)
	union = float(xr_max-xl_min)*float(yr_max-yl_min)

	iou = float(inter) / float(union)
	if iou < 0:
		iou = 0.0
	return iou
