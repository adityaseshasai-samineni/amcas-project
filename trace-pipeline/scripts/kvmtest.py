import fcntl,os
KVM_GET_API_VERSION=0xAE00
f=os.open("/dev/kvm",os.O_RDWR); print("KVM api version",fcntl.ioctl(f,KVM_GET_API_VERSION))
KVM_CREATE_VM=0xAE01
print("create VM fd",fcntl.ioctl(f,KVM_CREATE_VM,0))
