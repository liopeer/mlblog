+++
title = '2025 02 12_slurm_multipass'
date = 2025-02-12T21:48:54+01:00
draft = true
+++
## Installing Multipass
```bash
sudo snap install multipass
```
also worth noting: `microk8s` for potentional kubeflow cluster

## Create Subnet of the Host
this follows the official tutorial [Canonical – How to configure static IPs](https://canonical.com/multipass/docs/configure-static-ips)
```bash
sudo nmcli connection add type bridge con-name localbr ifname localbr ipv4.method manual ipv4.addresses 10.13.31.1/24
```
This command creates a new network bridge called localbr with a manually assigned static IP (10.13.31.1/24) using NetworkManager's command-line tool (nmcli).

Step-by-step breakdown:
sudo nmcli connection add → Adds a new network connection.
type bridge → The new connection is a network bridge.
con-name localbr → Names the new connection localbr (you can choose any name).
ifname localbr → Assigns the bridge interface the name localbr.
ipv4.method manual → Disables DHCP and assigns an IP manually.
ipv4.addresses 10.13.31.1/24 → Sets the static IP of the bridge to 10.13.31.1 with a subnet mask of /24 (255.255.255.0).

It creates a virtual network bridge named localbr, but does not add any network interfaces to it yet.
The bridge gets assigned a static IP address of 10.13.31.1/24.
This can be useful for Multipass, virtual machines, or containers that need to communicate over a virtual network.

Stop the bridge:
```bash
sudo nmcli connection down localbr
```
Launch it again:
```bash
sudo nmcli connection up localbr
```
Delete the bridge:
```bash
sudo nmcli connection delete localbr
```

## Launching the Main Node
```bash
multipass launch 22.04 --name=login01 --network="name=localbr,mode=manual"
```
Go into the node `multipass shell login01` and get the MAC address `ip addr show` and under the `link/ether` field.
Also in the node create and `vim` into:
`sudo vim /etc/netplan/10-custom.yaml`
and paste the following:
```bash
network:
    version: 2
    ethernets:
        extra0:
            dhcp4: no
            match:
                macaddress: "<MAC_ADRESS>"
            addresses: [10.13.31.50/24]
```
Instead of using `multipass shell` you can also use SSH to log into your VM. This requires
passing the private key set and accessing it requires ssh-ing as root:
`ssh -i /var/snap/multipass/common/data/multipassd/ssh-keys/id_rsa ubuntu@<INSTANCE_IP>`
The IP here is not the IP from above (this is for the `localbr` subnet). Instead it's the
one that's shown when you run `multipass list`.

## Launching of the Compute Nodes
```bash
multipass launch 22.04 --name=compute01 --network="name=localbr,mode=manual"
multipass launch 22.04 --name=compute02 --network="name=localbr,mode=manual"
multipass launch 22.04 --name=compute03 --network="name=localbr,mode=manual"
```

## Launching of the Centralized Storage
```bash
multipass launch 22.04 --name=storage --network name=localbr,mode=manual
```

## Assign Static IPs
