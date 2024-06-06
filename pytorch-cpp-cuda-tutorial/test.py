#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:test.py
# author:xm
# datetime:2024/6/6 15:10
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import torch
import cppcuda_tutorial

feats = torch.ones(2)
point = torch.zeros(2)

out = cppcuda_tutorial.trilinear_interpolation(feats, point)
print(out)
