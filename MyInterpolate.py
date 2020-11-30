import sys
EPSILON = sys.float_info.epsilon  # smallest possible difference

def print_list(values):
    print('[' + ', '.join(format(value, '.3f') for value in values) + ']')


def interpolate(inp, fi):
    i = int(fi)
    f = fi - i
    return (inp[i] if f < EPSILON else
            inp[i] + f*(inp[i+1]-inp[i]))

def singleArrayInterpolate(inp,new_len):

	delta = (len(inp)-1) / float(new_len-1)
	outp = [interpolate(inp, i*delta) for i in range(new_len)]
	
	print(len(inp))
	print(len(outp))
	return(outp)

