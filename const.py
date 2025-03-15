geoinfo = {
    "1la-":{ "lon": 110.335117, "lat": 19.995355, "start_index": 0 },
    "6mid":{"lon":110.332003,"lat":20.011541, "start_index": 5},
    "7l+":{"lon":110.345607,"lat":20.015613, "start_index": 10},
    "36la+":{"lon":110.337067, "lat":20.024766 , "start_index": 15},
    "61l-": {"lon":110.323076, "lat":20.013758, "start_index": 20},
}
lr = 0.000125
dim = 25
n = 15 # num_epochs
lm = [0.84, 0.99] # POT arg

# arg using for ablation
modelname = "GNSD" # GNSD or GNSDSingle
nsconst = 0
nsconst = 10e-100 # represent 0 when using GNSDSingle for avoid some unknown bug when nsconst = 0
nsconst = 20
