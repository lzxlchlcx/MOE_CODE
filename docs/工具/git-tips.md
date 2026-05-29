# Git 使用技巧

## 放弃本地，以远端为准

当本地和远端历史分叉，pull 后产生不想保留的 merge 时：

```bash
# 1. 中止正在进行的 merge
git merge --abort

# 2. 将本地强制重置为远端最新状态（丢弃所有本地提交和修改）
git reset --hard origin/master
```

### 适用场景
- 本地和远端历史不一致，pull 时触发了 merge
- 想直接用远端代码，不保留本地提交
- 合并冲突太多，想放弃合并重来

### 注意事项
- `git reset --hard` 会**永久丢弃**本地未推送的 commit 和未提交的修改，操作前确认远端已有你需要的内容
- 如果本地有需要保留的代码，先建分支备份：`git branch backup-xxx`
- `git merge --abort` 只能在 merge 进行中（冲突未解决/未提交）时使用

## 快捷提交

```bash
# 跳过 git add，直接提交所有已跟踪文件的修改
git commit -am "提交信息"

# 只提交某个文件
git commit 文件名 -m "提交信息"
```

- `-a` 自动暂存已跟踪文件的修改（不含 untracked 新文件）
- `git commit 文件名 -m "..."` 只提交指定文件，跳过 `git add`

## 常用快捷操作

### 查看类
```bash
# 查看简洁的提交历史（一行一条）
git log --oneline

# 查看最近 N 条提交
git log --oneline -5

# 查看某个文件的修改历史
git log --oneline 文件名

# 查看暂存区和工作区的差异
git diff

# 查看已暂存的改动
git diff --staged
```

### 撤销类
```bash
# 撤销工作区某个文件的修改（恢复到最近一次 commit 的状态）
git checkout -- 文件名

# 撤销暂存（unstage），文件修改保留在工作区
git reset HEAD 文件名

# 撤销最近一次提交，修改保留在工作区（不丢代码）
git reset --soft HEAD~1

# 撤销最近一次提交，修改保留在暂存区
git reset --mixed HEAD~1
```

### 分支类
```bash
# 创建并切换到新分支（合并 create + checkout）
git checkout -b 新分支名

# 切回上一个分支（不用记分支名）
git checkout -

# 删除已合并的本地分支
git branch -d 分支名

# 查看所有分支（含远端）
git branch -a
```

### 暂存类
```bash
# 临时保存当前工作区修改（不 commit）
git stash

# 查看暂存列表
git stash list

# 恢复最近一次暂存
git stash pop

# 恢复指定暂存（不删除记录）
git stash apply stash@{0}
```

### 其他
```bash
# 修改最近一次 commit 的提交信息
git commit --amend -m "新的提交信息"

# 查看某个文件的每一行最后是谁修改的
git blame 文件名

# 快速查看当前状态的一句话摘要
git status -s
```
