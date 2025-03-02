## Cloning & Pulling
```bash
git clone --recurse-submodules https://github.com/liopeer/mlblog.git
```
```bash
git pull --recurse-submodules
```

## Installing Hugo
```bash
brew install hugo
```

## Creating New Content
```bash
hugo new content content/posts/YYYY-MM-DD_<post-name>/index.md
```

## Including Jupyter Notebooks
```hugo
{{% ipynb notebook="<notebook-name>.ipynb" %}}
```
or for julia
```hugo
{{% ipynb_julia notebook="<notebook-name>.ipynb" %}}
```

## Building
```bash
hugo
```
Including drafts
```bash
hugo -D
```

## Serve Local
```bash
hugo server
```