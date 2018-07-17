class EarlyStopping():
	def __init__(self, patience=3):
		self._step = 0
		self._loss = float('inf')
		self.patience = patience

	def validate(self, loss):
		if self._loss < loss:
			self._step += 1
		if self._step >= self.patience:
			print('Early Stopping activated')
			return True

		else:
			self._step = 0
			self._loss = float('inf')
			return False

