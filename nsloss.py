import torch
import torch.nn as nn
from const import geoinfo

class Viscosity(nn.Module):
    def __init__(self, nu0, T0, S):
        super(Viscosity, self).__init__()
        self.nu0 = nu0
        self.T0 = T0
        self.S = S

    def forward(self, temperature_celsius):
        temperature_kelvin = temperature_celsius + 273.15
        nu = (self.nu0 * (self.T0 + self.S) / (temperature_kelvin + self.S)) * (temperature_kelvin / self.T0) ** 1.5
        return nu

class IdealGasDensity(nn.Module):
    def __init__(self, R, M):
        super(IdealGasDensity, self).__init__()
        self.R = R  
        self.M = M  

    def forward(self, p, T):
        T_kelvin = T + 273.15
        theoretical_rho = p * self.M / (self.R * T_kelvin)
        return theoretical_rho

class Finite(torch.nn.Module):
    def __init__(self, d=1e-5):
        super(Finite, self).__init__()
        self.d = d

    def forward(self, fx_d, fx_d_plus):
        f_prime_x = (fx_d_plus - fx_d) / (2 * self.d)        
        return f_prime_x
    
class SecondOrderFinite(torch.nn.Module):
    def __init__(self, d=1e-5):
        super(SecondOrderFinite, self).__init__()
        self.d = d

    def forward(self, fx_d, fx, fx_d_plus):
        f_double_prime_x = (fx_d_plus - 2 * fx + fx_d) / (self.d ** 2)
    
        return f_double_prime_x

class NSLoss(torch.nn.Module):
    def __init__(self):
        super(NSLoss, self).__init__()
        R = 8.314  # J/(mol·K)
        M = 0.02897  # kg/mol
        self.density = IdealGasDensity(R, M)
        
        nu0 = 1.458e-6  # m^2/s
        T0 = 288.15    # K
        S = 110.4      # Sutherland const, K
        self.u = Viscosity(nu0, T0, S)
    
        r = 6378.137*1000 # m
        
        self.prime_lat = Finite(d=(geoinfo['36la+']['lat'] - geoinfo['1la-']['lat'])*r/2)
        self.prime_lon = Finite(d=(geoinfo['7l+']['lon'] - geoinfo['61l-']['lon'])*r/2)
        self.prime_time = Finite(d=3600) 
        
        self.second_prime_lat = SecondOrderFinite(d=(geoinfo['36la+']['lat'] - geoinfo['1la-']['lat'])*r/2)
        self.second_prime_lon = SecondOrderFinite(d=(geoinfo['7l+']['lon'] - geoinfo['61l-']['lon'])*r/2)

    def forward(self, values): # values: (128,25)
        values = values.squeeze(0)
        d = self.density(values[:, 5]*1000, values[:, 7]) # kPa to Pa
        prime_p_y = self.prime_lat(values[:, 0]*1000, values[:, 15]*1000)
        prime_p_x = self.prime_lon(values[:, 20]*1000, values[:, 10]*1000)
        
        u = self.u(values[:, 7])
        
        speed_x = values[:, 8]
        speed_y = values[:, 9]
        
        u_1 = torch.roll(speed_x, shifts=1, dims=0)
        u_1[-1] = u_1[-2]
        u_2 = torch.roll(speed_x, shifts=-1, dims=0)
        u_2[0] = u_2[1]
        
        v_1 = torch.roll(speed_y, shifts=1, dims=0)
        v_1[-1] = v_1[-2]
        v_2 = torch.roll(speed_y, shifts=-1, dims=0)
        v_2[0] = v_2[1]
        
        prime_u_t = self.prime_time(u_2, u_1)
        prime_v_t = self.prime_time(v_2, v_1)
        
        prime_u_x = self.prime_lon(values[:, 23],values[:, 13])
        prime_u_y = self.prime_lat(values[:, 3], values[:, 18])
        prime2_u_x = self.second_prime_lon(values[:, 23], values[:, 8], values[:, 13])
        prime2_u_y = self.second_prime_lat(values[:, 3], values[:, 8], values[:, 18])
        
        prime_v_x = self.prime_lon(values[:, 24], values[:, 14])
        prime_v_y = self.prime_lat(values[:, 4], values[:, 19])
        prime2_v_x = self.second_prime_lon(values[:, 24], values[:, 9], values[:, 14])
        prime2_v_y = self.second_prime_lat(values[:, 4], values[:, 9], values[:, 19])
        
        Lx = d*(prime_u_t + speed_x*prime_u_x + speed_y*prime_u_y)+ prime_p_x - u*(prime2_u_x + prime2_u_y)
        Ly = d*(prime_v_t + speed_x*prime_v_x + speed_y*prime_v_y)+ prime_p_y - u*(prime2_v_x + prime2_v_y)
        return torch.mean(Lx**2 + Ly**2)

nsloss = NSLoss()
