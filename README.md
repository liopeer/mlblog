# Lio's ML Blog
A learning journal on machine learning, data science, and programming.
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

## Including Source Files (Jupyter Notebooks & Python Scripts)

### Jupyter Notebooks

```hugo
{{% ipynb notebook="<notebook-name>.ipynb" %}}
```
or for Julia:
```hugo
{{% ipynb_julia notebook="<notebook-name>.ipynb" %}}
```

### Python Scripts

You can include Python scripts in your posts using custom shortcodes:

- **Download only:**
  ```hugo
  {{< py_script_downloadonly script="myscript.py" >}}
  ```

- **Download and display inline:**
  ```hugo
  {{< py_script script="myscript.py" >}}
  ```

Make sure the script is added as a [Page Resource](https://gohugo.io/content-management/page-resources/) (e.g., placed in the same folder as your post's `index.md`).

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

## Updating the Forks
1. Make sure that the remotes are correctly configured: `git remote -v` and `origin` should point to the fork, while `upstream` should point to the original repo.
2. `git fetch --all` to update info on remotes
3. `git checkout master`
4. `git branch -vv` to check which remote branch `master` is tracking
5. `git branch --set-upstream-to=origin/master` in case it was wrong
6. `git merge upstream/master`
7. `git push`