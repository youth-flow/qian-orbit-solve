# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory contains reference PDFs by 钱学森 (Qian Xuesen) on aerospace and space navigation:

- `星际航行概论-钱学森-2008年版.pdf` — Introduction to Interstellar Navigation (2008 edition)
- `行星航行概论-钱学森-1963年版.pdf` — Introduction to Planetary Navigation (1963 edition)

## Usage with the create-review-materials skill

These PDFs serve as source material for generating Chinese LaTeX review materials (复习提纲, 练习题, 模拟卷). When asked to create study materials, use the `create-review-materials` skill with these PDFs as input.

## Reading PDFs

Use the Read tool with the `pages` parameter to read specific page ranges from these PDFs (max 20 pages per request). The files are large (~16-17 MB each), so always target specific sections rather than reading sequentially.
