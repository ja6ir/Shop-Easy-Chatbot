# Install the necessary libraries if you haven't done so
# !pip install faiss-cpu sentence-transformers pandas

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Step 1: Description of the App
app_description = """
Welcome to ShopEasy - an e-commerce app that simplifies online shopping. ShopEasy allows you to browse a wide variety of products, securely purchase items, and manage your orders effortlessly. Below is a step-by-step guide on how to use the app:

Step 1: Registration
- Open the ShopEasy app.
- Click on 'Sign Up' to create an account.
- Enter your email, create a password, and submit the form.
- Verify your email to complete the registration process.

Step 2: Browsing Products
- Use the search bar or browse by categories to explore products.
- You can filter products by price, rating, and other attributes.
- To see details, click on any product to open the product page.

Step 3: Adding Items to Cart
- On the product page, click 'Add to Cart' to add the product.
- You can review your cart at any time by clicking on the cart icon.

Step 4: Checkout and Payment
- Once you're ready to purchase, go to your cart and click 'Checkout.'
- Enter your shipping information and choose a payment method (credit card, PayPal, etc.).
- Review your order and click 'Place Order' to complete the purchase.

Step 5: Order Tracking and Returns
- After placing an order, you can track its status in the 'My Orders' section.
- If you need to return an item, go to 'My Orders,' select the order, and follow the instructions to initiate a return.

Step 6: Customer Support
- If you encounter any issues, ShopEasy's customer support is available via live chat or email.
"""

# Step 2: Create the full customer service dataset
data = {
    "Query": [
        "How do I reset my password?",
        "Can I change my shipping address after placing an order?",
        "What payment methods do you accept?",
        "How do I track my order?",
        "What is your return policy?",
        "How do I cancel my order?",
        "Can I update my payment method?",
        "How do I create an account?",
        "Can I view my order history?",
        "How do I change my email address?",
        "How do I delete my account?",
        "Can I leave a review on a product?",
        "Is there a guest checkout option?",
        "How do I apply a discount code?",
        "What are your shipping options?",
        "Can I combine multiple discount codes?",
        "How long does shipping take?",
        "How can I contact customer support?",
        "Do you offer free shipping?",
        "What should I do if I received a damaged item?",
        "Can I return an item after 30 days?",
        "What happens if I'm not home for delivery?",
        "How do I redeem my gift card?",
        "Are there bulk purchase discounts?",
        "How can I see new arrivals?",
        "Do you have a loyalty program?",
        "Can I set a password for my account?",
        "What is the difference between standard and expedited shipping?",
        "How do I know if an item is eligible for return?",
        "Can I leave a review without purchasing the product?",
        "How can I get the best deals?",
        "What if I encounter a technical issue with my account?",
        "Can I change my account password?",
        "Do you offer any warranties on your products?",
        "How do I contact a seller directly?",
        "Are there any hidden fees at checkout?",
        "Can I request a refund?",
        "Do you have a mobile payment option?",
        "How do I manage my notification preferences?",
        "What should I do if I receive a faulty product?",
        "How can I find my saved items?",
        "Can I purchase items on sale and still earn points?",
        "What should I do if I don’t receive my confirmation email?",
        "Can I use a personal coupon on top of a sale?",
        "Do you have an affiliate program?",
        "How do I report inappropriate content on your app?",
        "Can I see product reviews before purchasing?",
        "How do I find my order tracking link?",
        "Can I gift wrap multiple items in one order?",
        "What do I do if I accidentally placed a duplicate order?",
        "Are there any restrictions on shipping to certain countries?",
        "How do I unsubscribe from your newsletter?",
        "What do I do if I encounter a payment error?",
        "How long does it take to process a return?",
        "Can I exchange a product?",
        "How can I access my account settings?",
        "Can I save my payment information for future orders?",
        "Do you have a price match guarantee?",
        "How can I find out about upcoming sales?",
        "Can I change my shipping method after placing an order?",
        "What do I do if my credit card is charged but my order is not confirmed?",
        "How do I check if my order was successful?",
        "Do you offer seasonal discounts?",
        "How do I view my loyalty points balance?",
        "Can I share my account with someone else?",
        "What should I do if I can’t log in to my account?",
        "Are there any fees for using PayPal?",
        "How can I ensure my payment information is secure?",
        "Can I pre-order items that are not yet available?",
        "How do I change my account privacy settings?",
        "What if my gift card balance is not showing?",
        "Do you offer a price drop alert?",
        "How do I clear my app cache?"
    ],
    "Response": [
        "To reset your password, go to the login page and click on 'Forgot Password.' Follow the instructions sent to your registered email.",
        "Yes, you can change your shipping address within one hour of placing your order. Go to 'My Orders,' select the order, and update the address.",
        "We accept credit cards, debit cards, PayPal, and bank transfers. You can select your preferred method at checkout.",
        "You can track your order from the 'My Orders' section in the app or via the tracking link in your confirmation email.",
        "You can return most items within 30 days of receipt. Check the product page for specific return policies.",
        "Orders can be canceled within one hour of purchase. Go to 'My Orders' and select the order you wish to cancel.",
        "Yes, you can update your payment method before your order is confirmed in the 'Payment' section of checkout.",
        "To create an account, click 'Sign Up' on the app's login page and fill in the required details.",
        "Yes, you can view your order history under the 'My Orders' section in your account.",
        "To change your email address, go to 'Account Settings' and update your email information.",
        "To delete your account, please contact customer support, and they will assist you with account deletion.",
        "Yes, you can leave a review on any product you've purchased in the 'My Orders' section.",
        "Yes, we offer guest checkout for a quick purchase without creating an account.",
        "To apply a discount code, enter the code during checkout in the 'Discount Code' box.",
        "We offer standard, expedited, and next-day shipping options. You can choose at checkout.",
        "No, only one discount code can be applied per order.",
        "Standard shipping usually takes 5-7 business days, while expedited shipping is 2-3 business days.",
        "You can contact our customer support via the live chat option or by sending an email.",
        "Yes, we offer free shipping on orders over $50.",
        "If you received a damaged item, contact customer support within 48 hours for assistance.",
        "Unfortunately, we do not accept returns after 30 days from the date of receipt.",
        "If you're not home, the delivery company will leave a note or attempt redelivery.",
        "At checkout, enter your gift card code in the 'Gift Card' box and click 'Apply' to use it towards your purchase.",
        "Yes, bulk purchase discounts may apply. Please contact customer support for details on bulk orders.",
        "You can find new arrivals in the 'New Arrivals' section on the home page of the app.",
        "Yes, we have a loyalty program that rewards you with points for every purchase. You can redeem points for discounts on future orders.",
        "You set your password during account registration and can change it at any time in your account settings.",
        "Standard shipping is usually free and takes longer, while expedited shipping costs more and delivers faster.",
        "Check the product details on the item’s page. It will indicate if the item is returnable.",
        "No, only customers who have purchased the product can leave a review.",
        "Check our 'Deals' section regularly and subscribe to our newsletter for exclusive promotions.",
        "For technical issues, please report them through the app's feedback feature or contact customer support for assistance.",
        "Yes, you can change your password by going to 'Account Settings' and selecting 'Change Password.'",
        "Most products come with a warranty. Please check the product page for specific warranty information.",
        "You can contact a seller directly through the product page by clicking on 'Contact Seller.'",
        "No, there are no hidden fees at checkout. The total cost will be shown before you confirm your purchase.",
        "Yes, you can request a refund by contacting customer support within the return window.",
        "Yes, we offer mobile payment options like Apple Pay and Google Pay at checkout.",
        "Manage your notification preferences in the 'Settings' section under 'Notifications.'",
        "If you receive a faulty product, contact customer support within 48 hours for a resolution.",
        "To find saved items, go to your account and check the 'Saved Items' section.",
        "Yes, you can still earn loyalty points on sale items, but certain promotions may apply.",
        "If you don't receive your confirmation email, check your spam folder or contact customer support.",
        "Most personal coupons cannot be combined with sale items, but check the terms of each coupon.",
        "Yes, we offer an affiliate program where you can earn commissions for referring customers.",
        "You can report inappropriate content by clicking the 'Report' button next to the content.",
        "Yes, customer reviews are visible on each product page. They provide insights from other buyers.",
        "You can find your order tracking link in your order confirmation email or under 'My Orders.'",
        "To gift wrap items, select the 'Gift Wrap' option during checkout for each item.",
        "If you accidentally placed a duplicate order, contact customer support immediately for assistance.",
        "Shipping restrictions may apply to certain countries. Check our shipping policy for more details.",
        "To unsubscribe from the newsletter, click the 'Unsubscribe' link at the bottom of any email.",
        "If you encounter a payment error, double-check your payment details or contact your bank for assistance.",
        "Returns are typically processed within 7-10 business days after we receive the item.",
        "Yes, you can exchange products within the return window. Contact customer support for guidance.",
        "You can access your account settings by clicking on your profile icon in the app.",
        "Yes, you can save your payment information for future orders in the 'Payment Settings' section.",
        "Yes, we offer a price match guarantee if you find a lower price on an identical item.",
        "Upcoming sales are announced via our newsletter, so subscribe to stay informed.",
        "You cannot change your shipping method after placing an order, but you can contact customer support for assistance.",
        "If your credit card is charged but the order is not confirmed, please contact customer support for resolution.",
        "You can check your order status by going to 'My Orders' in the app.",
        "Yes, we offer seasonal discounts during major holidays and events.",
        "You can view your loyalty points balance in the 'Loyalty Program' section of your account.",
        "It's against our policy to share accounts, but you can invite others to create their own accounts.",
        "If you can't log in, use the 'Forgot Password' link to reset your password.",
        "We do not charge any fees for using PayPal, but check with PayPal for any applicable fees.",
        "Your payment information is secure with industry-standard encryption. We do not store your payment details.",
        "Yes, you can pre-order items not yet available. The expected release date will be shown on the product page.",
        "To change your privacy settings, go to 'Account Settings' and select 'Privacy Settings.'",
        "If your gift card balance is not showing, check with customer support to resolve the issue.",
        "Yes, you can sign up for price drop alerts on product pages to be notified of any price changes.",
        "To clear the app cache, go to your device settings, find the ShopEasy app, and select 'Clear Cache.'"
    ]
}

data['Query'].append("Can you describe the app's functionalities?")
data['Response'].append(app_description)
# Step 3: Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Step 4: Load a Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight transformer model for encoding

# Step 5: Encode the queries to get embeddings
query_embeddings = model.encode(df['Query'].tolist(), show_progress_bar=True)

# Step 6: Create a FAISS index and add the vectors
embedding_dimension = query_embeddings.shape[1]  # Get the dimension of the embeddings
index = faiss.IndexFlatL2(embedding_dimension)  # Create a FAISS index using L2 distance

# Step 7: Add embeddings to the index
index.add(np.array(query_embeddings).astype('float32'))

# Optional: Save the index to a file
faiss.write_index(index, 'faiss_index.bin')

# Optional: Save the DataFrame to a CSV for later reference
df.to_csv('customer_service_queries.csv', index=False)

print("FAISS index created and data loaded successfully!")


def search_vector_store(query, index, model, df, top_k=5):
    """
    Search the FAISS vector store for the most relevant queries.

    Parameters:
    - query (str): The input query to search for.
    - index (faiss.Index): The FAISS index to search in.
    - model (SentenceTransformer): The model used to encode the queries.
    - df (pd.DataFrame): The DataFrame containing queries and responses.
    - top_k (int): The number of top results to return.

    Returns:
    - list of tuples: A list of the top_k most relevant queries and their corresponding responses.
    """
    # Encode the user query
    query_embedding = model.encode([query]).astype('float32')
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the results
    results = []
    for i in range(top_k):
        results.append((df.iloc[indices[0][i]]['Query'], df.iloc[indices[0][i]]['Response']))
    
    return results



