import 'package:flutter/foundation.dart';
import 'package:url_launcher/url_launcher.dart';

class Utils {
  static Future openLink({
    @required String url,}) => _launchUrl(url);

  static Future _launchUrl(String url) async {
    if (await canLaunch(url)) {
      await launch(url);
    }
  }
  static Future openPhoneCall({@required String phoneNumber}) async {
    final url = 'tel:$phoneNumber';
    await _launchUrl(url);
  }
}