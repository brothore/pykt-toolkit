
# 安装常用插件
echo "正在安装插件..."
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone --depth=1 https://github.com/marlonrichert/zsh-autocomplete ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autocomplete

# 安装Powerlevel10k主题
echo "正在安装Powerlevel10k主题..."
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# 迁移bash设置到zshrc
echo "迁移bash配置到zshrc..."
grep -E '^(export|PATH|alias|#)' ~/.bashrc >> ~/.zshrc
echo -e "\n# 以下设置由bashrc迁移而来" >> ~/.zshrc

# 添加用户指定的关键配置
echo "添加自定义配置到zshrc..."
cat >> ~/.zshrc << 'EOL'

# --- 用户指定配置 ---
source /etc/profile
source /etc/autodl-motd
export PYTHONPATH="/root/autodl-tmp/pykt-toolkit/:$PYTHONPATH"
# -------------------

# 插件设置
plugins=(
  git
  zsh-autosuggestions
  zsh-syntax-highlighting
  zsh-autocomplete
  docker
  sudo
  copyfile
  history
)

# 主题设置
ZSH_THEME="powerlevel10k/powerlevel10k"


echo "安装完成！请执行后续手动操作"