# EDITO Hello Process

This is a minimal Dockerized batch job for EDITO-style testing.

It writes a single text file to an S3/MinIO bucket:

- default key: `hello-process-test/hello.txt`
- default content: `hello world`

## Files

- `app.py`: writes the text file to S3
- `Dockerfile`: container image entrypoint
- `requirements.txt`: Python dependency list

## Required environment variables

- `AWS_BUCKET_NAME`
- `AWS_S3_ENDPOINT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Optional environment variables

- `AWS_SESSION_TOKEN`
- `OUTPUT_PREFIX` default: `hello-process-test`
- `OUTPUT_FILE_NAME` default: `hello.txt`
- `HELLO_MESSAGE` default: `hello world`

## Local Docker test

Build:

```bash
docker build -t edito-hello-process .
```

Run:

```bash
docker run --rm \
  -e AWS_BUCKET_NAME=project-moi-ai \
  -e AWS_S3_ENDPOINT=minio.dive.edito.eu \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_SESSION_TOKEN=... \
  -e OUTPUT_PREFIX=hello-process-test \
  -e OUTPUT_FILE_NAME=hello.txt \
  -e HELLO_MESSAGE="hello world" \
  edito-hello-process
```

## Using it on EDITO

This example is intended for the EDITO contribution flow where you deploy from a Dockerfile or from a built image.

Use these environment variables in the EDITO form:

- `AWS_BUCKET_NAME`
- `AWS_S3_ENDPOINT`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
- `OUTPUT_PREFIX`
- `OUTPUT_FILE_NAME`
- `HELLO_MESSAGE`

Suggested values:

- `OUTPUT_PREFIX=hello-process-test`
- `OUTPUT_FILE_NAME=hello.txt`
- `HELLO_MESSAGE=hello world`

Expected result:

```text
s3://<AWS_BUCKET_NAME>/hello-process-test/hello.txt
```

with content similar to:

```text
hello world
created_at_utc=2026-03-13T12:34:56.000000+00:00
```
