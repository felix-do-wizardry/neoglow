# %%
import websocket
import json, time, string
import numpy as np
import os

# %%
def vector_norm(v):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    return v / np.max([np.sqrt(np.sum(v ** 2)), 0.000001])

class PixelSet:
    def __init__(self, pos=[0, 0, 0]):
        self.pos = np.array(pos, dtype=int)
        self.count = len(pos)

class PixelSet_Strip(PixelSet):
    def __init__(self,
                pos_origin=[0, 0, 0],
                pos_center=None,
                vector=[1, 1, 1],
                spacing=16.666667,
                count=60,
                ):
        self.vector = vector_norm(vector)
        if pos_center is not None:
            self.pos_origin = np.array(pos_center) - self.vector * spacing * (count - 1) / 2
        else:
            self.pos_origin = np.array(pos_origin)
        pos = self.pos_origin[None] + self.vector[None] * spacing * np.arange(count)[:, None]
        super().__init__(pos=pos)

class PixelSet_Frame(PixelSet):
    def __init__(self,
                pos_origin=[0, 0, 0],
                vectors=[[1, 0, 0], [0, 1, 0]],
                spacing=16.666667,
                size=[12, 6],
                corner_offset=0.,
                ):
        self.vectors = np.array([vector_norm(v) for v in vectors])
        _dot = np.dot(*self.vectors)
        assert _dot == 0
        
        self.pos_origin = np.array(pos_origin)
        self.size = np.array(size)
        assert tuple(self.size.shape) == tuple([2])
        
        self.corner_offset = corner_offset
        frame_lengths = np.array([v + self.corner_offset * 2 for v in self.size])
        poss = []
        for i in range(4):
            side_offsets = frame_lengths * [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ][i]
            side_offset = np.sum(side_offsets[:, None] * self.vectors, axis=0)
            side_vector = self.vectors[i % 2] * (1 if i <= 1 else -1)
            _pos = self.pos_origin[None] + (side_offset + side_vector[None] * np.arange(self.size[i % 2])[:, None]) * spacing
            poss.append(_pos)
        
        pos = np.concatenate(poss, axis=0)
        super().__init__(pos=pos)

class PixelSet_Matrix_Zigzag(PixelSet):
    def __init__(self,
                pos_origin=[0, 0, 0],
                vectors=[[1, 0, 0], [0, 1, 0]],
                spacing=16.666667,
                size=[32, 8],
                # corner_offset=0.,
                ):
        self.vectors = np.array([vector_norm(v) for v in vectors])
        _dot = np.dot(*self.vectors)
        assert _dot == 0
        
        self.pos_origin = np.array(pos_origin)
        self.size = np.array(size)
        assert tuple(self.size.shape) == tuple([2])
        
        poss = []
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                _col = col
                if row % 2 == 1:
                    _col = self.size[1] - 1 - col
                poss.append(np.sum(self.vectors * [[_col], [row]], axis=0))
        pos = np.stack(poss, axis=0)
        pos = pos * spacing + self.pos_origin
        
        super().__init__(pos=pos)


# %%
# ps_stand = PixelSet_Frame(
#     pos_origin=[1200, 100, 900],
#     vectors=[[1, 0, 0], [0, 1.8, 1]],
#     spacing=1000/60,
#     size=[12, 8],
#     corner_offset=1.,
# )
# ps_stand.pos.shape

# %%
# ps_m32x8 = PixelSet_Matrix_Zigzag(
#     pos_origin=[1500, 400, 750],
#     vectors=[[0, -1, 0], [1, 0, 0]],
#     spacing=1000/100,
#     size=[32, 8],
# )
# ps_m32x8.pos.shape

# %%
def float2hex(v=1.0, digit=2, fill=True):
    _v = int(v * 16 ** digit)
    _v = min(max(_v, 0), 16 ** digit - 1)
    s = f'{_v:x}'[-digit:]
    if fill:
        s = s.rjust(digit, '0')
    return s

def int2hex(v=1, digit=2, fill=True):
    _v = int(min(max(v, 0), 16 ** digit - 1))
    s = f'{_v:x}'[-digit:]
    if fill:
        s = s.rjust(digit, '0')
    return s


# %%
class NeoWS:
    def __init__(self, ip='192.168.3.100', port=8908):
        self.ip = ip
        self.port = port
        self.address = f'ws://{ip}:{port}'
        # self.ws = websocket.WebSocket()
        self.connect()
    
    def connect(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(self.address)
    
    def send(self,
                offset,
                time_duration=2000,
                hue_base=0.,
                hue_delta=1.,
                saturation=1.,
                value=0.2,
                ):
        
        self.ws.send('|'.join([
            ''.join([int2hex(v, 4) for v in offset]),
            float2hex(hue_base, 3),
            float2hex(hue_delta, 3),
            float2hex(saturation, 2),
            float2hex(value, 2),
            int2hex(time_duration, 10),
        ]))

# %%
class NeoControl:
    def __init__(self, ip_prefix='192.168.3', ip_values=[], pixel_pos=[], port=8908):
        self.port = port
        self.ip_prefix = ip_prefix
        assert isinstance(ip_values, list)
        # assert isinstance(neows, list)
        # assert len(neows) == len(ip_values)
        # self.ip_values = ip_values
        
        self.count = 0
        self.ips = []
        self.neows = []
        self.pixel_pos = []
        
        _ips = [f'{self.ip_prefix}.{v}' for v in ip_values]
        for _ip, _pixel_pos in zip(_ips, pixel_pos):
            self.add_neows(ip=_ip, pixel_pos=_pixel_pos)
    
    def add_neows(self, ip, pixel_pos):
        assert isinstance(ip, str)
        # assert isinstance(pixel_set, PixelSet)
        assert isinstance(pixel_pos, np.ndarray)
        if ip in self.ips:
            return False
        self.neows.append(NeoWS(ip=ip, port=self.port))
        self.pixel_pos.append(pixel_pos)
        self.ips.append(ip)
        self.count = len(self.neows)
        return True
    
    def send(self,
                pos2hue_fn,
                # offset,
                fn_kwargs={},
                time_duration=2000,
                # hue_base=0.,
                # hue_delta=1.,
                # saturation=1.,
                # value=0.2,
                **kwargs,
                ):
        
        # assert callable(pos2hue_fn)
        
        for i in range(self.count):
            _velocity = 1e-12
            # _duration = 2000.
            # offset_stand = set_stand.pos[:, 0] / _velocity * 1000
            
            _offset = pos2hue_fn(
                self.pixel_pos[i],
                time_duration=time_duration,
                **fn_kwargs,
            )
            
            self.neows[i].send(
                offset=_offset,
                time_duration=time_duration,
                # hue_base=0,
                # hue_delta=1,
                # saturation=1,
                # value=1.,
                **kwargs,
            )

# %%
tv_mid_x = 700
tv_y = 120
desk_z = 750
ps_edge = PixelSet_Strip(
    pos_center=[600, 0, desk_z],
    vector=[-1, 0, 0],
    spacing=1000/60,
    count=60,
)
ps_tv_bottom = PixelSet_Strip(
    pos_center=[tv_mid_x, tv_y, desk_z + 80],
    vector=[-1, 0, 0],
    spacing=1000/144,
    count=144,
)
ps_tv_top = PixelSet_Strip(
    pos_center=[tv_mid_x, tv_y, desk_z + 700],
    vector=[1, 0, 0],
    spacing=1000/60,
    count=60,
)
ps_edge_tv = PixelSet(
    pos=np.concatenate([
        ps_edge.pos,
        ps_tv_bottom.pos,
        ps_tv_top.pos,
    ], axis=0),
)
ps_edge_tv.pos.shape

# %%
NC = NeoControl(
    ip_prefix='192.168.3',
    ip_values=[101],
    # ip_values=[103],
    pixel_pos=[ps_edge_tv.pos],
    # pixel_pos=[ps_edge.pos],
    # pixel_pos=[ps_m32x8.pos],
    port=8908,
)
NC

# %%
def get_cycle(vector=[1, 0, 0], cycle_length=600, time_duration=1000):
    def cycle(pos, **kwargs):
        # _velocity = kwargs.get('velocity', 30)
        # _time_duration = kwargs.get('time_duration', 2000)
        return ((pos @ vector_norm([vector]).reshape(-1, 1) / cycle_length) * time_duration)
        # return ((pos @ vector_norm([[1, 0.6, 0]]).reshape(-1, 1) / _velocity) * 1000) % _time_duration
    
    return cycle

_velocity = 120
_cycle_length = 1600
_time_duration = _cycle_length / _velocity * 1000
# _time_duration = 20_000

NC.send(
    # get_cycle([1, 0, 0]),
    get_cycle(
        # vector=[1, 1, -0.7],
        vector=[1, 0, 0],
        cycle_length=_cycle_length,
        time_duration=_time_duration,
    ),
    fn_kwargs=dict(
        # velocity=2000,
        # velocity=100,
        # velocity=1_000_000 / _time_duration * 2,
    ),
    # time_duration=12000,
    time_duration=_time_duration,
    hue_base=0.,
    hue_delta=1.,
    saturation=1.,
    # value=0.25,
    value=0.5,
)

# %%
import plotly.express as px
import plotly.graph_objects as go

# %%
_pos = ps_edge_tv.pos
fig = go.Figure(data=[go.Scatter3d(
    x=_pos[:, 0],
    y=_pos[:, 1],
    z=_pos[:, 2],
    mode='markers',
    marker=dict(
        size=8,
        color=np.arange(_pos.shape[0]),
        # colorscale='Viridis',   # choose a colorscale
        opacity=1
    )
)])

fig.add_trace(
    go.Mesh3d(
        x=[0, 0, 1200, 1200],
        y=[0, 600, 0, 600],
        z=[750] * 4,
        # colorbar_title='z',
        colorscale=[[0, 'rgb(200, 200, 200)'], [1, 'rgb(200, 200, 200)']],
        # colorscale=[[0, 'gold'],
        #             [0.5, 'mediumturquoise'],
        #             [1, 'magenta']],
        # Intensity of each vertex, which will be interpolated and color-coded
        intensity=[0] * 4,
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        # i=[0, 0, 0, 1],
        # j=[1, 2, 3, 2],
        # k=[2, 3, 1, 3],
        name='desk_1',
        showscale=False,
    )
)

fig.add_trace(
    go.Mesh3d(
        x=[v + 1220 for v in [0, 0, 1200, 1200]],
        y=[0, 600, 0, 600],
        z=[750] * 4,
        colorscale=[[0, 'rgb(50, 50, 50)'], [1, 'rgb(50, 50, 50)']],
        intensity=[0] * 4,
        name='desk_0',
        showscale=False,
    )
)

_max_value = max(np.max(_pos) * 1.1, 2500)
fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0, _max_value]),
        yaxis = dict(nticks=4, range=[0, _max_value]),
        zaxis = dict(nticks=4, range=[0, _max_value]),
        aspectmode='cube',
    ),
    # yaxis=dict(scaleanchor="x", scaleratio=1),
    # zaxis=dict(scaleanchor="x", scaleratio=1),
    template='plotly_dark',
    margin=dict(l=0, r=0, b=0, t=0),
    width=800,
    height=800,
)
fig.show()

# %%
def ping(ip, delay=0.2):
    res = os.system(f"ping -c 1 -W {delay:.3f} {ip} > /dev/null 2>&1")
    return res == 0

ping('192.168.3.101')


# %%
time_start = time.time()
print()
for i in range(1_000_000):
    time_elapsed = time.time() - time_start
    if time_elapsed >= 12:
        break
    response = os.system("ping -c 1 -W 0.2 " + '192.168.3.101' + " > /dev/null 2>&1")
    _ping = ping('192.168.3.101')
    print(f'\rtime[{time_elapsed:.1f}s] ping[{response == 0}]' + ' ' * 20, end='')
    time.sleep(0.2)

print()

# %%
neos = {
    v: NeoWS(ip=f'192.168.3.{v + 100}', port=8908)
    # for v in [101, 103][:]
    for v in [3]
}

# %%
# velocity in mm/s | speed/duration in ms |
# _velocity = 200.
# _speed = 6000.
_velocity = 1e-12
_velocity = 800
_duration = 4000.
offset_stand = ps_stand.pos[:, 0] / _velocity * 1000
offset_edge = ps_edge.pos[:, 0] / _velocity * 1000

np.min([offset_edge.min(), offset_stand.min()])

# neos[1].send(
#     offset=set_stand.pos[:, 0] / _velocity * 1000,
#     time_duration=_duration,
#     hue_base=0,
#     hue_delta=1,
#     saturation=1,
#     value=1.,
# )
neos[1].send(
    offset=ps_edge.pos[:, 0] / _velocity * 1000,
    time_duration=_duration,
    hue_base=0,
    hue_delta=1,
    saturation=1,
    value=1.,
)

# %%
float2hex(0.05, digit=3)
int2hex(42, digit=3)

# %%
ws = websocket.WebSocket()
# ws.connect("ws://192.168.3.101:8908")
# ws.connect("ws://192.168.3.102:8908")
ws.connect("ws://192.168.3.101:8908")

_count = 40
time_duration = 4000
# offset = [i / _count * time_duration for i in range(_count)]
offset = [np.abs(i / (max(_count, 2) - 1) * 2 - 1) * time_duration for i in range(_count)]
hue_base = 0.
hue_delta = 1.
saturation = 1.
value = 0.2

ws.send('|'.join([
    ''.join([int2hex(v, 4) for v in offset]),
    float2hex(hue_base, 3),
    float2hex(hue_delta, 3),
    float2hex(saturation, 2),
    float2hex(value, 2),
    int2hex(time_duration, 10),
]))
# result = ws.recv()
# print("Received: ", result)

# %%
_count = 40
time_duration = 4000
# offset = [i / _count * time_duration for i in range(_count)]
offset = np.zeros([_count], dtype=float)
offset[:12] = np.arange(12) + 1
offset[12:20] = [12 + 1] * 8
offset[20:32] = 12 - np.arange(12)
offset[32:] = [0] * 8
offset

offset = offset / 24 * time_duration

hue_base = 0.
hue_delta = 0.6
saturation = 1.
value = 0.2

ws.send('|'.join([
    ''.join([int2hex(v, 4) for v in offset]),
    float2hex(hue_base, 3),
    float2hex(hue_delta, 3),
    float2hex(saturation, 2),
    float2hex(value, 2),
    int2hex(time_duration, 10),
]))