def feature_extraction_step(net, batch, batch_idx, **kw):
	inputs, _ = batch
	inputs = inputs.to(kw['device'])
	features = net(inputs)

	return {'predictions':features}

