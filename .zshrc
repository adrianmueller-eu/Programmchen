export ZSH="$HOME/.oh-my-zsh"
export ZDOTDIR=$ZSH
export PATH="/opt/homebrew/bin:$PATH"
export FPATH="$(brew --prefix)/share/zsh/site-functions:$FPATH"
export FPATH="$(brew --prefix)/share/zsh-completions:$FPATH"

HIST_STAMPS="dd.mm.yyyy" # see man strftime
SAVEHIST=10000000

plugins=(
  git
  npm
  fd
  conda-zsh-completion
  fast-syntax-highlighting
)

source "$ZSH/oh-my-zsh.sh"

# export ARCHFLAGS="-arch x86_64" # compilation flag
# export SSH_KEY_PATH="~/.ssh/rsa_id"
# setopt sh_word_split

source ~/.profile

### completions
COMPLETION_WAITING_DOTS="true"
ENABLE_CORRECTION="true"
autoload -U +X bashcompinit && bashcompinit
#autoload -U +X compinit && compinit
#_comp_options+=(
#  globdots # show hidden folders in tab completion without preceding "."
#)

# https://stackoverflow.com/questions/14307086/tab-completion-for-aliased-sub-commands-in-zsh-alias-gco-git-checkout
if hash pass &>/dev/null; then
  compdef pc=pass
  compdef pe=pass
  compdef pm=pass
  compdef pu=pass
  compdef pi=pass
  compdef po=pass
fi
compdef _path_commands fip
compdef _functions fif
compdef _aliases fia
compdef '_files -g "*.tex"' clat
compdef '_files -g "*.tex"' cltex
compdef '_files -g "*.md"' md2h
compdef '_files -g "*.ipynb"' j

### theme
ZSH_THEME_GIT_PROMPT_PREFIX=" %{$fg[red]%}"
ZSH_THEME_GIT_PROMPT_SUFFIX="%{$reset_color%}"
ZSH_THEME_GIT_PROMPT_DIRTY="%{$fg[yellow]%}*"
# disable marking untracked files under VCS as dirty
# DISABLE_UNTRACKED_FILES_DIRTY="true"
ZSH_THEME_GIT_PROMPT_CLEAN=""
# see http://zsh.sourceforge.net/Doc/Release/Prompt-Expansion.html
PROMPT='%(?..%F{red})%*%f %c$(git_prompt_info)%(!.%F{red}.)$%f '

### zsh things
alias t="whence -avs"
unalias gsd

function command_not_found_handler() {
  if type command_not_found >/dev/null 2>&1; then
    command_not_found "$@"
  else
    echo "Sorry, couldn't find: $*"
  fi
}

function help() {
  case $1 in
  test)
    man -P "less -p'^CONDITIONAL EXPRESSIONS$'" zshall
    ;;
  "")
    man zshall
    ;;
  *)
    man -P "less -p'^ {7}$@ '" zshall
    ;;
  esac
}

pride() (
  # https://codegolf.stackexchange.com/questions/230438/create-a-pride-flag
  i(){echo "\e[$1m${(pl:22::█:)}"};i 31;i 91;i 93;i 32;i 34;i 35
)
alias rainbow=pride

#for IntelliJ shell do
#echo "source ~/.zshrc" >> /Applications/IntelliJ\ IDEA.app/Contents/plugins/terminal/.zshrc
# If zsh is slow, try
# mv ~/.oh-my-zsh/lib/spectrum.zsh ~/.oh-my-zsh/lib/spectrum.zsh_disabled

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/homebrew/Caskroom/miniconda/base/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        . "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    else
        export PATH="/opt/homebrew/Caskroom/miniconda/base/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
