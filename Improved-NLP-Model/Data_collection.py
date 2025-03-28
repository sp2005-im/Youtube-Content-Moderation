class Youtube_Metadata_Collector():
  def __init__(self,categories,max_results,youtube):
    self.categories = categories 
    self.max_results = 50
    self.youtube = youtube

  def get_video_ids(self, search_terms):
    # Collect video ids for a given category and search terms
    video_ids = []

    for term in search_terms:
      request = self.youtube.search().list(
          part="id",
          maxResults=self.max_results,
          q=term,
          type="video" #Ensures only video results
      )
      response = request.execute()
      for item in response.get('items', []):  
        if 'id' in item and 'videoId' in item['id']:  
          video_ids.append(item['id']['videoId'])
    return video_ids[:self.max_results]  # Limit to max_results per category
  
  def get_video_details(self,video_id):
    """Get metadata for a specific video"""
    video_response = self.youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    ).execute()

    if not video_response['items']:
        return None

    video_data = video_response['items'][0]
    snippet = video_data['snippet']
    statistics = video_data.get('statistics', {})
    '''
    # Get comments if available
    comments = []
    try:
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=10
        ).execute()

        for item in comments_response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except:
        # Comments might be disabled
        pass
    '''
    return {
        'video_id': video_id,
        'title': snippet.get('title', ''),
        'description': snippet.get('description', ''),
        'channel_title': snippet.get('channelTitle', ''),
        'publish_date': snippet.get('publishedAt', ''),
        'view_count': statistics.get('viewCount', 0),
        'like_count': statistics.get('likeCount', 0),
        'category': None  # Will be filled later
    }

  def clean_text(self,text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text 
  
  def preprocess_text(self, text):
    """Full text preprocessing pipeline"""
    # Clean text
    text = self.clean_text(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)
  
  # Main data collection function
  def collect_youtube_data(self):
    all_videos = []

    for category, search_terms in self.categories.items():
        print(f"Collecting videos for category: {category}")
        video_ids = self.get_video_ids(search_terms)

        for video_id in video_ids:
            video_data = self.get_video_details(video_id, search_terms)
            if video_data:
                video_data['category'] = category
                all_videos.append(video_data)

    return pd.DataFrame(all_videos)
