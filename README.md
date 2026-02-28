# DadOps Blog Code

Verified, runnable code from [DadOps blog posts](https://daddaops.github.io/dadops-site/blog.html).

Every code example in our blog posts has been extracted, tested, and verified on real hardware. No fabricated benchmarks — only real measurements.

## Structure

Each blog post has its own directory:

```
post-slug/
  README.md          # What this code does + link to blog post
  requirements.txt   # Python dependencies
  *.py               # Runnable scripts
  output/            # Captured output, benchmarks, logs
```

## Running

```bash
cd post-slug/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 script_name.py
```
