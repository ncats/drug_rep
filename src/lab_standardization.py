
		
		

	
	
def get_24_multiplier(rate):
	per_day_pattern = re.compile(r'\bper 24 hours')
	match = per_day_pattern.search(rate)
	if match: return 24
