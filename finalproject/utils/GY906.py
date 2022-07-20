#!/usr/bin/python
#class for read gy906
import smbus2
class GY906(object):

    MLX90614_RAWIR1=0x04
    MLX90614_RAWIR2=0x05
    MLX90614_TA=0x06
    MLX90614_TOBJ1=0x07
    MLX90614_TOBJ2=0x08
    MLX90614_TOMAX=0x20
    MLX90614_TOMIN=0x21
    MLX90614_PWMCTRL=0x22
    MLX90614_TARANGE=0x23
    MLX90614_EMISS=0x24
    MLX90614_CONFIG=0x25
    MLX90614_ADDR=0x0E
    MLX90614_ID1=0x3C
    MLX90614_ID2=0x3D
    MLX90614_ID3=0x3E
    MLX90614_ID4=0x3F

    def __init__(self, address=0x5a, bus_num=1, units = "c"):
        self.bus_num = bus_num
        self.address = address
        self.bus = smbus2.SMBus(bus=bus_num)
        self.units = units

    def read_reg(self, reg_addr):
        try :
            return self.bus.read_word_data(self.address, reg_addr)
        except:
            return None

    def pass_c(self, celsius):
        return celsius - 273.15

    def pass_k(self, celsius):
        return celsius

    def pass_f(self, celsius):
        return (celsius - 273.15) * 9.0/5.0 + 32

    def data_to_temp(self, data):
        temp = (data*0.02)
        temperature = getattr(self, "pass_" + self.units)(temp)
        return temperature

    def get_amb_temp(self):
        data = self.read_reg(self.MLX90614_TA)
        if data != None:
            return self.data_to_temp(data)
        else:
            return None

    def get_obj_temp(self):
        data = self.read_reg(self.MLX90614_TOBJ1)
        if data != None:
            return self.data_to_temp(data)
        else:
            return None