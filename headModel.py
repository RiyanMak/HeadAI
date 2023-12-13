

def headModel(initialInput, feedBack1, feedBack2, feedBack3, feedBack4, feedBack5, feedBack6, feedBack7, feedBack8, toggle):
  if(toggle): #toggle toggles recursive mode, which first inputs, then sends the work back to models
      model = SentenceTransformer('all-MiniLM-L6-v2')
      inputSummarization1 = summarizer(feedBack1, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization2 = summarizer(feedBack2, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization3 = summarizer(feedBack3, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization4 = summarizer(feedBack4, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization5 = summarizer(feedBack5, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization6 = summarizer(feedBack6, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization7 = summarizer(feedBack7, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      inputSummarization8 = summarizer(feedBack8, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
      embeddings1 = np.array(model.encode(inputSummarization1))
      embeddings2 = np.array(model.encode(inputSummarization2))
      embeddings3 = np.array(model.encode(inputSummarization3))
      embeddings4 = np.array(model.encode(inputSummarization4))
      embeddings5 = np.array(model.encode(inputSummarization5))
      embeddings6 = np.array(model.encode(inputSummarization6))
      embeddings7 = np.array(model.encode(inputSummarization7))
      embeddings8 = np.array(model.encode(inputSummarization8))

      embed_avg = (embeddings1 + embeddings2 + embeddings3 + embeddings4 + embeddings5 + embeddings6 + embeddings7 + embeddings8)/8
      cosine1 = np.dot(embed_avg,embeddings1)/(norm(embed_avg)*norm(embeddings1))
      cosine2 = np.dot(embed_avg,embeddings2)/(norm(embed_avg)*norm(embeddings2))
      cosine3 = np.dot(embed_avg,embeddings3)/(norm(embed_avg)*norm(embeddings3))     
      cosine4 = np.dot(embed_avg,embeddings4)/(norm(embed_avg)*norm(embeddings4))
      cosine5 = np.dot(embed_avg,embeddings5)/(norm(embed_avg)*norm(embeddings5))
      cosine6 = np.dot(embed_avg,embeddings6)/(norm(embed_avg)*norm(embeddings6))    
      cosine7 = np.dot(embed_avg,embeddings7)/(norm(embed_avg)*norm(embeddings7))
      cosine8 = np.dot(embed_avg,embeddings8)/(norm(embed_avg)*norm(embeddings8))

      cosinelist = [cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8]
      max_val = max(cosineList)
      idx_max = cosineList.index(max_val)

  else:
      #send prompt over to the models
