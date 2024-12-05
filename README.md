# IBM-Data-Science-Capstone-SpaceY

## Introduction
In this capstone, we will predict if the SpaceX Falcon 9 first stage will land successfully. As a data scientist working for SpaceY competing against SpaceX for a bid, determing the cost of a launch for a successful landing of the first stage will be my goal. And for this, I committed myself to gathering essential information about SpaceX and making relevant dashboards for my team.

![SpaceX Launch Failure Instances](https://github.com/debdattasarkar/SpaceX-Data-Science-Project/blob/master/images/fail.gif)
![SpaceX Launch Success Instances](https://github.com/debdattasarkar/SpaceX-Data-Science-Project/blob/master/images/success.gif)

### Background
SpaceX, a leader in the space industry, strives to make space travel affordable for everyone. Its accomplishments include sending spacecraft to the international space station, launching a satellite constellation that provides internet access and sending manned missions to space. SpaceX advertises Falcon 9 rocket launches on its website, with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because SpaceX can reuse the first stage. Therefore, if we can determine if the first stage will land, we can determine the cost of a launch. To do this, public data, and machine learning models to predict whether SpaceX can reuse the first stage are used.

### Questions to address
* What factors of a mission influence Falcon 9 launch success?
* How payload mass, launch site, orbit type, and other features affect first stage landing success?
* What condtions does SpaceX have to meet to get the best results and ensure the best successful first stage landing rates?
* Best predictive model for successful first stage landing.

## Methodology
The research attempts to identify the factors for a successful rocket landing. To make this determination, the following methodologies where used:
* **Data Collection** using SpaceX REST API and web scraping techniques
* **Data Wrangling** to handle missing values and transform variables for further analysis
* **Exploratory Data Analysis** with:
  * Data visualization techniques for analyzing trend and patterns
  * SQL for understanding launch statistics related to past outcomes, and other key factors
* **Interactive Visual Analytics** using:
  * Folium for interactive maps, calculates distances between launch sites to its proximities, and also helps assess the geographical terrain of a site
  * Plotly Dash for dashboard, gives total success launches for a site, and also gives correlation between payload and launch outcome for a specific payload range 
* **Predictive Modeling** to predict the best performing model out of logistic regression, support vector machine (SVM), decision tree and K-nearest neighbor (KNN)

## Results

### Exploratory Data Analysis Insights
* Launch success has improved over time
* KSC LC-39A has the highest success rate among all launch sites
* Orbits ES-L1, GEO, HEO, and SSO have a 100% success rate

### Data Analytics
* Most launch sites are near the equator.
* The launch sites are in close proximity to the coastline and have a good logistic infrastructure around, aiding transportation of materials and people

### Predictive Analytics
* All models generally performed well, as three of the models had the same classification accuracy of 83%
* The best performing model out of the four was Decision Tree with an accuracy of 94%.
* The confusion matrix was evaluated with FN = 5, FP = 1, TN = 0, and TP = 12 for the following metrics and its results are shown below:
  * Precision = 0.92
  * Recall = 0.70
  * F1-Score = 0.79
  * Accuracy = 0.66

## Conclusion
* **Best Performed Model:** Decision Tree classifier
* **Payload mass:** Across all launch sites, higher the payload mass, higher were the launch success rates
* **Launch Success:** Increased over the years
* **Launch sites to proximities:** All launch sites are near coastline and not quite far away in terms of accessing the railways and highways
* **Equator closeness:** The launch sites being close to equator were able to get an additional natural boost - due to the rotational speed of earth - which helps   save the cost of putting in extra fuel and boosters too 
* **KSC LC-39A:** Has the highest success rate among all launch sites. Also, has a 100% success rate for launches carrying payloads less than 5,500 kg 
* **Orbits:** ES-L1, GEO, HEO, and SSO have a 100% success rate, while SO has a 0% success rate

## Future Scope and Directions
* **Expand Dataset:** Large and diverse dataset ensures consistency and reliability across varied scenarios
* **Advanced Feature Analysis / PCA:** This helps identify the most influential features, reduce noise and improve accuracy, thereby retaining critical         
  information both performance and interpretability can be enhanced
* **XGBoost:** Is a powerful model which can outperform the four models used here for modeling. This model helps increase robustness and reduce overfitting
* **Real-time and Operational Enhancements:** Integrate real-time weather and telemetry data for dynamic outcome prediction. Also, accommodate an automated     
  system, integrating predictive results with visual analytics to optimize launch configurations
* **Extend Predictive Capabilities:** Use additional metrics like fuel efficiency, and RNNs or Transformers for time-series analysis
* **Ethical and Sustainability Considerations:** Incorporating features like carbon footprint per launch can be considered to optimize rocket reuse and minimize 
  environmental impact  
