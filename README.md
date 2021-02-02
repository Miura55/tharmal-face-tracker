# tharmal-face-tracker
Tharmal face tracking systeme using Open VINO.

## Setup

### Setup Open VINO
1. Install Toolkit from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/)

2. Create folder from this command
```
sudo mkdir -p /opt/intel/openvino
```

3. Unpack toolkit
```
sudo tar -xf  INSTALL_DIR/l_openvino_toolkit_runtime_raspbian_p_<version>.tgz --strip 1 -C /opt/intel/openvino
```

4. Install cmake
```
sudo apt install cmake
```

5. Set the environ veriables
```
source /opt/intel/openvino/bin/setupvars.sh
```

Set `.bashrc`
```
echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
```

6. Add User `users` group
```
sudo usermod -a -G users "$(whoami)"
```

7. Install USB rules
```
sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```

### Install Library

```
pip3 install seaborn
```

### Activate pigpio

```
sudo systemctl enable pigpiod
```
