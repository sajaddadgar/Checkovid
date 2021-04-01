from django.db import models

# Create your models here.
class Claim(models.Model):
    text = models.TextField()
    verdict = models.CharField(max_length=1)
    vector = models.TextField()