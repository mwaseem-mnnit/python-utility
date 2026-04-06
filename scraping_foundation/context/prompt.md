
Below i have provided the task and expectation from you

The task is to:
- Start from all product list page URL with ?page=1 till page=16
- for each page iterate on product grid element is <ul id="product-grid">
- for each item in the <ul> tag, visit the link represented by <a href="/products/<product-url>">
- from the product info page we need to retreive below information and write to a csv 
- generate an product id for, format p_<6 digit number  start at 100001> and add to colum product_id
- extract product slug from url and add to column product_slug in the csv, the url format is <bas_url>/products/<product_slug>
- download the images into a folder using the <slide-component id="GalleryViewer-template--17166756642988__main"> tag, the name of the images should be <product_id>_<index starting from 0 for each product> and add it to the csv comma separated
- title: the <h1> tag inside the div <div class="product__title">, add to title column in the csv
- description: in the div <div class="product__description rte quick-add-hidden">
	- there are multiple paragraph tags, some are bold using <strong>
	- use column names as desc_1, desc_2 etc, these all <p> contains description, specification etc of the product

- Throttle and rate limit:
	- support to iterate just for one product for testing the flow
	- support to wait before visiting the next product info page
	- before visitng a page first check the csv, if the product slug exist, ignore

Generate a sequence of cursor prompt for above task, break the task into below steps and provide a sequence of cursor prompts in coding format to implement those
Task1:
	- create a csv for the result with all the columns mentioned leaving the description column which will be dynamic
	- define classes/types/objects that will represent a row in the csv 
	- a util function to generate product id, image file names etc
	- a funtion to check if a product aready exists in the csv
	- a function to wait based on the configurd environment variable

Task2:	
	- a function to download html content, parse in such a manner that we can read it for our purpose
	- main worker function which will use the created function to acheve the task

Use python programming language