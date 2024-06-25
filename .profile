#############################
### paths
#############################
export P=$HOME/Programmchen
export PATH="$P:$PATH"
alias p="cd $P"

export BREW_PREFIX=$(brew --prefix)
export GOROOT=$BREW_PREFIX/opt/go/libexec
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8) # /Library/Java/JavaVirtualMachines/openjdk.jdk/Contents/Home
export ANDROID_SDK_ROOT="$BREW_PREFIX/share/android-sdk"
export ANDROID_HOME=$ANDROID_SDK_ROOT
export PYTHONSTARTUP=$HOME/.ipython/profile_default/startup/00.py
export HOMEBREW_NO_ENV_HINTS=1

export PATH="$BREW_PREFIX/sbin:$PATH"
export PATH="$BREW_PREFIX/opt/python/libexec/bin:$PATH" # current python version # @3.9
export PATH="./node_modules/.bin:$PATH" # nice hack for dev independent node deps versioning
#export PATH="$JAVA_HOME/bin:$PATH"

if cat ~/.nanorc &>/dev/null | grep -e '$BREW_PREFIX' &>/dev/null; then
  sed -i "" "s/\$BREW_PREFIX/${BREW_PREFIX//\//\\/}/" ~/.nanorc
fi

#############################
### prefs
#############################
export LC_ALL='en_US.UTF-8'
#export LANG=C
export EDITOR="$BREW_PREFIX/bin/mate -w"
export SPELL="aspell -c"
export NO_GETTEXT=1 # disable localization support and make git use english only
export CONDA_AUTO_ACTIVATE_BASE=false
export PAGER=less

#############################
### interactive shell start
#############################
if [[ $- == *i* ]]; then
  #case `date +%d%m` in
  #  2412|2512)
  #    shXmas
  #    ;;
  #esac
  fortune -a
fi

# update hosts file
update_hosts_after=$((14*24*60*60 + $RANDOM)) # seconds
if [[ ! -e /etc/hosts  # not exists
  || $(wc -l /etc/hosts | awk '{print $1}') -lt 100  # too short
  || ! $(tail -1 /etc/hosts | cut -c 3-) =~ ^[0-9]{10}$  # timestamp not exists
  || $(tail -1 /etc/hosts | cut -c 3-) -lt $(($(date +%s) - update_hosts_after))  # timestamp expired
]]; then
  echo
  echo "Time to update hosts file!"
  # fetch into tmphosts
  tmphosts="${TMPDIR}hosts"
  curl --connect-timeout 3 --max-time 30 -# --output "$tmphosts" https://raw.githubusercontent.com/StevenBlack/hosts/master/alternates/fakenews-gambling-porn/hosts
  if [[ $? -eq 0 ]]; then
    # append new timestamp
    echo "# $(date +%s)" >> "$tmphosts"
    # show updates
    if [[ -f /etc/hosts ]]; then
      PAGER= git diff --shortstat /etc/hosts "$tmphosts"
    else
      echo "Where is your hosts file? Creating a new one!"
    fi
    allowed=($(git config --global hosts.allowed))
    for link in "${allowed[@]}"; do
      sed -i "" -r "/[\. ]($link)/d" "$tmphosts"
    done
    if [[ ! -w /etc/hosts ]]; then
      echo "Your password is required to write the hosts file."
    fi
    if [[ -f /etc/hosts ]]; then
      sudo mv /etc/hosts /etc/hosts_backup || echo "Couldn't create a backup!"
    fi
    sudo chown root "$tmphosts"
    sudo mv "$tmphosts" /etc/hosts && echo "Update succeeded!" || echo "Update failed!"
    sudo -k # invalidate cached credentials
    unset allowed link
  else
    echo "Update failed (couldn't download)!"
  fi
fi

#############################
### source custom
#############################
if [[ $0 == "bash" ]]; then
  export SHELL=/bin/bash
fi
sources=(
  "$HOME/.public_profile" # should be first
  "$HOME/.private_profile"
  "$HOME/.fzf.$(basename $SHELL)"
  "$HOME/.config/broot/launcher/bash/br"
)

for f in "${sources[@]}"; do
  source "$f"
done

return;
#############################
### help section
#############################

#############################
### OS
#############################

# Start
startup=(
/Library/LaunchAgents
/Library/LaunchDaemons
/Library/PrivilegedHelperTools
/Library/Apple/System/Library/LaunchDaemons
~/Library/LaunchAgents
)
for d in "${startup[@]}"; do
  echo "$d:"
  l "$d"
done

# File recovering
# try this (https://forum.cgsecurity.org/phpBB3/viewtopic.php?t=7727):
# Due to macOS High Sierra (macOS 10.13) requirement, Mac users are not allowed an access to the built-in system drive from any apps. Therefore, if you need to restore lost data from the system disk under macOS 10.13, please "disable System Integrity Protection" first.
# How to disable "disable System Integrity Protection"? Please follow the steps below.
# Step 1: Reboot the Mac and hold down"Command + R" keys simultaneously after you hear the startup chime, this will boot OS X into Recovery Mode.
# Step 2: When the "OS X Utilities" screen appears, pull down the "Utilities" menu at the top of the screen instead, and choose "Terminal".
# Step 3: In the "Terminal" window, type in "csrutil disable" and press "Enter" then restrart your Mac.

#############################
### shell
#############################

# uncomment if git autocompletion is too slow
#function __git_files () { _wanted files expl 'local files' _files  }

### Some other interesting things
# set -- ab bc bd	set positional parameters $1 $2 ...
## $
# $?	last command exit code
# $-	???
# $$	process ID of the shell
## cd
# cd	go to home dir (~)
# cd -	go to last dir ($OLDPWD)
## VARIABLES
# PWD		current dir
# OLDPWD	last dir
# ~-		shorthand for $OLDPWD
# LINENO	number of current command in current shell
## bash4
# &>	shorthand for 2>&1>
# ;&	in case statement: fallthrough
# ;;&	in case statement: evaluate "through"
## zsh
# n	go to n-th last dir, where n \in {1,2,...,9}
# -	1 (see above)
# $zsh_eval_context	see https://zsh.sourceforge.io/Doc/Release/Parameters.html#Parameters-Set-By-The-Shell
## sed
# :  # label
# =  # line_number
# a  # append_text_to_stdout_after_flush
# b  # branch_unconditional
# c  # range_change
# d  # pattern_delete_top/cycle
# D  # pattern_ltrunc(line+nl)_top/cycle
# g  # pattern=hold
# G  # pattern+=nl+hold
# h  # hold=pattern
# H  # hold+=nl+pattern
# i  # insert_text_to_stdout_now
# l  # pattern_list
# n  # pattern_flush=nextline_continue
# N  # pattern+=nl+nextline
# p  # pattern_print
# P  # pattern_first_line_print
# q  # flush_quit
# r  # append_file_to_stdout_after_flush
# s  # substitute
# t  # branch_on_substitute
# w  # append_pattern_to_file_now
# x  # swap_pattern_and_hold
# y  # transform_chars

# bash is weird: IFS
IFS="i "
a1="ii"
for b in $a1; do echo "b: [$b]"; done
# []
# []
a2="  " # expect: two empty strings
for b in $a2; do echo "b: [$b]"; done
a3="i "
for b in $a3; do echo "b: [$b]"; done
# []

#############################
### git history
#############################

#help for bfg --replace-text:
#PASSWORD1                       # Replace literal string 'PASSWORD1' with '***REMOVED***' (default)
#PASSWORD2==>examplePass         # Replace with 'examplePass' instead
#PASSWORD3==>                    # Replace with the empty string
#regex:password=\w+==>password=  # Replace, using a regex
#regex:\r(\n)==>$1               # Replace Windows newlines with Unix newlines
#https://stackoverflow.com/questions/4110652/how-to-substitute-text-from-files-in-git-history
#after that use `git maintenance run`

#replace git commit name & mail address:
function replace_git_name_and_mail() {
  printf "Have you replaced the default with the mail address you want to be replaced? [y]: "
  read trash
  printf "Did you make a copy of the repository before doing this? [y]: "
  read trash

  git filter-branch -f --env-filter '
    OLD_EMAIL="your-old-email@example.com"
    CORRECT_NAME="your-name"
    CORRECT_EMAIL="your-new-email@example.com"
    if [[ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]]; then
      export GIT_COMMITTER_NAME="$CORRECT_NAME"
      export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
    fi
    if [[ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]]; then
      export GIT_AUTHOR_NAME="$CORRECT_NAME"
      export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
    fi
  ' --tag-name-filter cat -- --branches --tags
  echo "All occurrences replaced!"
  echo "See if you have to remove backup branches: git update-ref -d refs/original/refs/heads/master"
  echo "After that do: git rebase --root && git reflog expire --expire=now --all && git gc --prune=now --aggressive"
}
#https://stackoverflow.com/questions/750172/how-to-change-the-author-and-committer-name-and-e-mail-of-multiple-commits-in-gi

# use for specific commit edit
# git rebase --interactive --root
# then, for author change (committer?): git commit -a --amend --author="someone <someone@example.com>"
# https://stackoverflow.com/questions/1186535/how-to-modify-a-specified-commit

# grep through all git history
# PAGER= git grep <regexp> $(git rev-list --all)
