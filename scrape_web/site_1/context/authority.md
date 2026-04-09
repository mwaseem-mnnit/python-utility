## Authority:
Information of the website:
- the all product link is: https://hjgstore.com/collections/all-hjg-liu-hjg-products-at-hjgstore-com
  - product grid element is <ul id="product-grid"> ...
  - the grid is paginated, the pagination element is <ul class="pagination__list list-unstyled">
  - for each card there is an href link to product main page
  - in the home page there are below elements
- <div class="product__title">: represents product title
- <div class="product__description rte quick-add-hidden">: represent parent product desscription
	- this have many child <p> elements, containing detailed description, specification etc
  - The product images are list item of the list tag: <ul id="Slider-Thumbnails-template--<epoch>__main" class="thumbnail-list list-unstyled slider slider--mobile slider--tablet-up">
    - each list item have img src tag: <img src="//hjgstore.com/cdn/shop/files/HJG7InchMinusTriProjectorMain-cmpr.webp?v=1774271558&amp;width=416" ... >
    - the first one should be downloaded as thumbnail as well as product info, with suffix _thmb
    - rest of the item should be downloaded as product info images with index, for each product start index from 0
  - <slide-component id="GalleryViewer-template--17166756642988__main">: highlighted image on the product info page
  - <slide-component id="GalleryThumbnails-template--17166756642988__main">: slider to all the images of this product, we need to use this to download all related images for this product
      - <ul id="Slider-Thumbnails-template--17166756642988__main">: inside the above slider component we have this list element, each item in this list contains an image.





