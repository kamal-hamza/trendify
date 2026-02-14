from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings


@shared_task
def add(x, y):
    """Example task that adds two numbers."""
    return x + y


@shared_task
def mul(x, y):
    """Example task that multiplies two numbers."""
    return x * y


@shared_task
def send_email_task(subject, message, recipient_list):
    """
    Send an email asynchronously.
    
    Args:
        subject: Email subject
        message: Email body
        recipient_list: List of recipient email addresses
    """
    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=recipient_list,
        fail_silently=False,
    )
    return f"Email sent to {len(recipient_list)} recipient(s)"


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
)
def fetch_rss_feeds(self):
    """
    Fetch and process RSS feeds.
    This task will automatically retry up to 3 times with 60 seconds between retries.
    """
    try:
        # TODO: Implement RSS feed fetching logic
        print(f"Fetching RSS feeds - Task ID: {self.request.id}")
        return "RSS feeds fetched successfully"
    except Exception as exc:
        print(f"Error fetching RSS feeds: {exc}")
        raise


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=5,
)
def process_trending_data(self):
    """
    Process trending data with exponential backoff retry.
    Uses exponential backoff with jitter for retries.
    """
    try:
        # TODO: Implement trending data processing logic
        print(f"Processing trending data - Task ID: {self.request.id}")
        return "Trending data processed successfully"
    except Exception as exc:
        print(f"Error processing trending data: {exc}")
        raise


@shared_task
def cleanup_old_data(days=30):
    """
    Clean up old data from the database.
    
    Args:
        days: Number of days to keep data (default: 30)
    """
    from datetime import timedelta
    from django.utils import timezone
    
    cutoff_date = timezone.now() - timedelta(days=days)
    
    # TODO: Implement cleanup logic for your models
    print(f"Cleaning up data older than {cutoff_date}")
    
    return f"Cleaned up data older than {days} days"