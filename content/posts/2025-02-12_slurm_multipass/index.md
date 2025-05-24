+++
title = '2025 02 12_slurm_multipass'
date = 2025-02-12T21:48:54+01:00
draft = true
+++
## Launching an LDAP Instance
```bash
cd ldap
docker compose -f docker-compose.yaml up -d
```
Add gidNumber, uidNumber, homeDirectory to the user schema and gidNumber to the group schema.

Access it at [http://localhost:17170](http://localhost:17170) and add your users. This largely follows [Medium – A new LDAP contender: LightLDAP](https://blog.raduzaharia.com/a-new-ldap-contender-lightldap-0452aa0baee9). You can also use [Google Workspace](https://www.linkedin.com/pulse/linux-sssd-google-workspace-harman-s-kapoor-05pve/).

We will now need some Unix specific attributes.

## Installing Multipass
```bash
sudo snap install multipass # some comment
```
also worth noting: `microk8s` for potentional kubeflow cluster

## Creating a Bridge Network
Current network config (`/etc/netplan/01-network-manager-all.yaml`) probably looks like this:
```yaml
# Let NetworkManager manage all devices on this system
network:
  version: 2
  renderer: NetworkManager
```
We need to change it to:
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eno1:
      dhcp4: no
  bridges:
    br0:
      interfaces: [eno1]
      dhcp4: yes
```
You might have to enable systemd-networkd:
```bash
sudo systemctl enable systemd-networkd.service
sudo systemctl start systemd-networkd.service
```
And then apply the changes:
```bash
sudo netplan apply
```


## Launching the Main Node
```bash
multipass launch 22.04 --name=login01 --network="name=br0"
```
You can now set a static IP with your router.

Instead of using `multipass shell` you can also use SSH to log into your VM. This requires
passing the private key set and accessing it requires ssh-ing as root:
`ssh -i /var/snap/multipass/common/data/multipassd/ssh-keys/id_rsa ubuntu@<INSTANCE_IP>`
The IP here is not the IP from above (this is for the `localbr` subnet). Instead it's the
one that's shown when you run `multipass list`.

### Connecting it to LDAP
Inside the VM, install SSSD:
```bash
sudo apt install sssd sssd-tools
```
Then configure it to use LDAP.
```bash
# /etc/sssd/sssd.conf
[sssd]
domains = lldap
services = nss, pam

[domain/lldap]
id_provider = ldap
auth_provider = ldap
access_provider = permit

ldap_uri = ldap://localhost:3890
ldap_search_base = dc=example,dc=com

ldap_schema = rfc2307bis
ldap_user_object_class = inetOrgPerson
ldap_group_object_class = groupOfNames
ldap_user_home_directory = /home/%u

# Optional tuning
ldap_referrals = false
enumerate = true
cache_credentials = true
```