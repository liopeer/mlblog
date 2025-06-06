build:
	hugo

build-drafts:
	hugo -D

build-serve-drafts:
	hugo -D
	hugo server -D --bind=0.0.0.0

build-serve:
	hugo
	hugo server --bind=0.0.0.0