/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package webcrawler;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 *
 * @author apurv
 */
public class WebCrawler {
//   HashSet<String> links;
//   public WebCrawler(){
//        links = new HashSet<String>();
//       }
//   public void getPageLinks(String url){
//       if(!links.contains(url))
//       {
//           try
//           {
//               if(links.add(url))
//                   System.out.println(url);
//               Document document = Jsoup.connect(url).get();
//               Elements linksonpage = document.select("a[href]");
//                for (Element page : linksonpage) {
//                    getPageLinks(page.attr("abs:href"));
//                }
//           }
//           catch(Exception e)
//           {
//               
//           }
//       }
//   }

    public static void main(String[] args) throws IOException {

//        new WebCrawler().getPageLinks();
//        Document doc = Jsoup.connect("http://codeforces.com/contest/779/submission/25053611").get();
//        Elements ele = doc.select("a[href]");
//        for (Element e : ele) {
//            System.out.println(e.attr("href"));
//            System.out.println(e.text());
//        }
//        System.out.println("");
//        
//        Elements codes = doc.getElementsByClass("prettyprint lang-pl program-source");
//        for(Element src : codes)
//        {
//            //System.out.println(src.getElementsByTag("spam").);
//        }
        ArrayList<String> arr = new ArrayList<>();
        Document doc = Jsoup.connect("http://codeforces.com/contest/779/status/C").get();
        Elements ele = doc.select("a[href].view-source");
        String ele2 = doc.getElementsByClass("id-cell dark left").text();
        System.out.println(ele2);
        ele.forEach((Element e) -> {   
            boolean te;
            te = arr.add(e.attr("href"));
        });
         System.out.println(arr);
         String s = "http://codeforces.com";
        for(int i = 0 ; i < arr.size() ;i++)
        {
            Document doce = Jsoup.connect(s+arr.get(i)).get();
            String temp = doce.getElementsByTag("pre").first().text();
            System.out.println(temp);
            
            System.out.println("\n\n\n\n\n\n\n\n\n\n");
        }
        /*for (Element e : temp) {
                System.out.println(e.getElementsByTag("span").first().text());
        }*/
    }
}
