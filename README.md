<p align="center">
  <img src="https://github.com/liopeer/mlblog/actions/workflows/format_ruff.yaml/badge.svg" alt="Ruff Formatter"/>
  <img src="https://github.com/liopeer/mlblog/actions/workflows/format_julia.yaml/badge.svg" alt="Julia Formatter"/>
</p>

<h1 align="center">Lio's ML Blog</h1>
<p align="center">
  <em>A learning journal on machine learning, data science, and programming.</em>
</p>

---

## 🚀 Quick Start

### Clone with Submodules

```bash
git clone --recurse-submodules https://github.com/liopeer/mlblog.git
```

If you already cloned without `--recurse-submodules`, initialize submodules with:
```bash
git submodule update --init --recursive
```

### Pull Latest Changes (including submodules)

```bash
git pull --recurse-submodules
git submodule update --remote
```

---

## 🗂️ Submodules & Forks

This repo uses [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for themes and dependencies.

- **Personal fork (e.g., PaperMod):**  
  - `origin` points to your fork.
  - `upstream` points to the original repo.
- **External submodules:**  
  - Only `origin` (the main repo) is needed.

**Check remotes:**
```bash
git remote -v
```

**For submodules that are forks:**
```bash
cd themes/PaperMod
git remote add upstream https://github.com/adityatelange/PaperMod.git  # if not already set
git fetch upstream
```

---

## 🔄 Updating Submodules

**Update all submodules to latest remote:**
```bash
git submodule update --remote
```

**For a forked submodule (e.g., PaperMod):**
```bash
cd themes/PaperMod
git fetch upstream
git checkout master
git merge upstream/master
git push origin master
```

**For external submodules:**
```bash
cd path/to/submodule
git checkout master
git pull origin master
```

---

## 🛠️ Installing Hugo

```bash
brew install hugo
```

---

## ✍️ Creating New Content

```bash
hugo new content/posts/YYYY-MM-DD_<post-name>/index.md
```

---

## 📓 Including Jupyter Notebooks

```hugo
{{% ipynb notebook="<notebook-name>.ipynb" %}}
```
or for Julia:
```hugo
{{% ipynb_julia notebook="<notebook-name>.ipynb" %}}
```

---

## 🏗️ Building the Site

```bash
hugo
```
To include drafts:
```bash
hugo -D
```

---

## 🔥 Local Development

```bash
hugo server
```

---

## 🧭 Tips for Working with Forks & Upstreams

1. **Check remotes:**  
   `git remote -v`  
   - `origin` should be your fork.
   - `upstream` should be the original repo (for forked submodules).

2. **Fetch all remotes:**  
   `git fetch --all`

3. **Switch to master/main:**  
   `git checkout master`

4. **Check tracking branch:**  
   `git branch -vv`

5. **Set upstream if needed:**  
   `git branch --set-upstream-to=origin/master`

6. **Merge upstream changes:**  
   `git merge upstream/master`

7. **Push to your fork:**  
   `git push`

---

## 📚 Resources

- [Git Submodules Guide](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Hugo Documentation](https://gohugo.io/documentation/)