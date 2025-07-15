#!/bin/bash
#拉取原本的pykt库的更新，然后更新到自己fork的版本的main分支

# 切换到main分支并更新
git checkout main
git fetch upstream
git merge --ff-only upstream/main
git push origin main

# 返回原始分支
git checkout try_unbalance