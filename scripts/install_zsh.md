安装和美化 Zsh 的完整指南
1. 安装 Zsh

sudo apt install zsh

设置 Zsh 为默认 shell
chsh -s $(which zsh)

2. 安装 Oh My Zsh (Zsh 配置框架)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
3. 安装常用插件
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search

git clone https://github.com/agkozak/zsh-z ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-z


4. 配置插件
编辑 ~/.zshrc文件：

nano ~/.zshrc
找到 plugins=(git)这一行，修改为：

plugins=(
  git
  zsh-syntax-highlighting
  zsh-autosuggestions
  zsh-history-substring-search
  zsh-z
)
5. 安装主题 (Powerlevel10k)
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

ZSH_THEME="powerlevel10k/powerlevel10k"
6. 应用配置
source ~/.zshrc
7. 配置 Powerlevel10k
首次运行时会提示配置向导，按照提示进行配置即可。如果你错过了，可以运行：

p10k configure
8. 可选: 安装 Nerd Fonts
为了获得最佳图标显示效果，建议安装 Nerd Fonts：

# 例如安装 FiraCode Nerd Font
brew tap homebrew/cask-fonts
brew install --cask font-fira-code-nerd-font
然后在终端设置中使用该字体。

9. 其他实用插件 (可选)
自动跳转项目目录
git clone https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/autojump ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/autojump
快速切换目录
git clone https://github.com/rupa/z.git ~/.zsh/z
10. 更新 Oh My Zsh 和插件
定期更新以获得最新功能：

omz update