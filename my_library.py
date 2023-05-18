def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def test_it():
  return 'loaded'
  
  
def cond_probs_product(table, evidence_values, target_column, target_val):
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns [:-1]
  evidence_complete = up_zip_lists (evidence_columns, evidence_values) 
  cond_prob_list = []
  for evidence_column, evidence_val in evidence_complete:
    cond_prob_value = cond_prob (table, evidence_column, evidence_val, target_column, target_val)
    cond_prob_list += [cond_prob_value]
  partial_numerator = up_product (cond_prob_list)
  return partial_numerator
