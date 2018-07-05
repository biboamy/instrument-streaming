ENCODER = {
	'res_block':
	{
		'conv': {'ksize': (3,3), 'padding': (1,1)},
		'pooling': {'ksize': (3,1), 'stride': (3,1), 'padding': (1,0), 'return_indices': True}
	},
	'layer1': {'inp': 1, 'oup': 32},
	'layer2': {'inp': 32, 'oup': 64},
	'layer3': {'inp': 64, 'oup': 128},
	'layer4': {'inp': 128, 'oup': 256},
	'gru': {'inp': 512, 'oup': 256, 'stack_num': 1, 'batch_first': True, 'bidirectional': True}
}

PITCHDECODER = {
	'TConv1': {'inp': 256, 'oup': 128, 'ksize': (3,1), 'stride': (3,1), 'padding': (1,0)},
	'BatchNorm1': {'num':128},
	'TConv2': {'inp': 128, 'oup': 64, 'ksize': (3,1), 'stride': (3,1), 'padding': (1,0)},
	'BatchNorm2': {'num':64},
	'TConv3': {'inp': 64, 'oup': 32, 'ksize': (3,1), 'stride': (3,1), 'padding': (0,0)},
	'BatchNorm3': {'num':32},
	'TConv4': {'inp': 32, 'oup': 1, 'ksize': (3,1), 'stride': (3,1), 'padding': (1,0)},
	'pooling': {'ksize': (1,2), 'stride': (1,2)}
}

INSTDECODER = {
	'TConv1': {'inp': 256, 'oup': 128, 'ksize': (3,1), 'stride': (1,1), 'padding': (1,0)},
	'BatchNorm1': {'num':128},
	'TConv2': {'inp': 128, 'oup': 64, 'ksize': (3,1), 'stride': (3,1), 'padding': (1,0)},
	'BatchNorm2': {'num':64},
	'TConv3': {'inp': 64, 'oup': 32, 'ksize': (3,1), 'stride': (1,1), 'padding': (1,0)},
	'BatchNorm3': {'num':32},
	'TConv4': {'inp': 32, 'oup': 1, 'ksize': (3,1), 'stride': (3,1), 'padding': (1,0)},
	'pooling': {'ksize': (1,2), 'stride': (1,2)}
}