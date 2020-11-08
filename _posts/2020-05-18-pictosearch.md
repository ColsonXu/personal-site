---
title: 'PictoSearch Development - Idea'
date: 2020-05-18 00:00:00
description: Motivation and thought process behind PictoSearch, an AI powered camera text searching tool.
featured_image: '/images/pictosearch/Diagram.png'
tags: ["ios", "pictosearch"]
---

## MOTIVATION
As I was studying for my introductory business law class in my school’s library, I struggled to find a term amid pages of cluttered text. Maybe you know that feeling too: that word has to be in there, but I just can’t find it! A thought struck my mind that only if there is a tool that let me search through non-digital text just as I would use Ctrl-F on my computer. What a great idea! There has to be someone who had the same thought and made an app for it. However, when I try to find one, nothing. Most apps that claim to be able to search through documents are basically just OCR scanners and the user will then be able to search within the recognized text. We all know how unreliable OCR works and most apps involving OCR requires users to pay a fee to use it. I want something that can recognize text in real-time and highlight the keywords I am searching for. Time to design my own app.

![Finding a word in an ocean of text.](/images/pictosearch/Reading.jpeg)

At a high level, this app shouldn't be very complex. I can use a pre-existing machine learning platform like TensorFlow from Google or CoreML from Apple to deal with the heavy lifting.

![The overarching architecture of this app.](/images/pictosearch/Diagram.png)

After extensive research, I have decided to first start with Apple's CoreML kit since it is easier to implement and works really well with iPhone's hardware acceleration.

To be continued...

