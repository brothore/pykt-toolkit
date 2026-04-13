cp /root/autodl-tmp/pykt-toolkit/scripts/id_ed25519.pub ~/.ssh/id_rsa.pub

cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

chmod 600 ~/.ssh/authorized_keys 

chmod 700 ~/.ssh 

