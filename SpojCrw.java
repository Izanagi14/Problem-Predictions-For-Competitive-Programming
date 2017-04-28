/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package webcrawler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 *
 * @author apurv
 */
public class SpojCrw {

    public static void main(String[] args) throws IOException {

        Document doc1 = Jsoup.connect("http://www.spoj.com/ranks/users/").get();
        ArrayList<String> arr = new ArrayList<String>();
        Element e1 = doc1.select("table").get(0);
        Elements el1 = e1.getElementsByTag("a");
        for (Element el : el1) {
            arr.add(el.attr("href").toString());
        }
        int temp[] = new int[105];
        Arrays.fill(temp, 3);
        TreeMap<String, Vector<Integer>> ts = new TreeMap<String, Vector<Integer>>();
        for (int ii = 0; ii < 5; ii++) {
            Document doc = Jsoup.connect("http://www.spoj.com" + arr.get(ii)).get();
            System.out.println(arr.get(ii));
            for (int i = 0; i <= 1; i++) {
                Element ele = doc.select("table").get(i);
                Elements rows = ele.select("tr");

                for (Element e : rows) {
                    String s = e.getElementsByTag("td").text();
                    for (String sep : s.split(" ")) {
                        ts.put(sep, new Vector<Integer>());
                    }
                }
            }
        }
        Vector<TreeMap<String, Integer>> Mat1 = new Vector<TreeMap<String, Integer>>();
        for (int ii = 0; ii < 5; ii++) {
            Document doc = Jsoup.connect("http://www.spoj.com" + arr.get(ii)).get();
            System.out.println(arr.get(ii));
            TreeMap<String, Integer> Mat = new TreeMap<String, Integer>();
            for (int i = 0; i <= 1; i++) {
                Element ele = doc.select("table").get(i);
                Elements rows = ele.select("tr");

                for (Element e : rows) {
                    String s = e.getElementsByTag("td").text();
                    for (String sep : s.split(" ")) {
                        if (i == 0) {
                            Mat.put(sep, 1);
                        } else {
                            Mat.put(sep, 0);
                        }
                    }
                }
            }
            Mat1.add(Mat);
        }

        Iterator<Map.Entry<String, Vector<Integer>>> it = ts.entrySet().iterator();

        TreeMap<String, Vector<Integer>> finalans = new TreeMap<String, Vector<Integer>>();

        while (it.hasNext()) {
            Map.Entry<String, Vector<Integer>> entry = it.next();
            Vector<Integer> vec = new Vector<Integer>();
            //System.out.println(entry);
            for (int i = 0; i < 5; i++) {
                int flag = 0;

                TreeMap<String, Integer> gy = Mat1.get(i);
                if (gy.containsKey(entry.getKey())) {
                    vec.add(gy.get(entry.getKey()));
                    finalans.put(entry.getKey(), vec);
                } else {
                    vec.add(3);
                    finalans.put(entry.getKey(), vec);
                }

            }
        }
        Iterator<Map.Entry<String, Vector<Integer>>> ti = finalans.entrySet().iterator();
        while (ti.hasNext()) {
            Map.Entry<String, Vector<Integer>> en = ti.next();
            System.out.println(en.getKey() + " " + en.getValue());

        }
    }
}
//                Iterator<Map.Entry<String, Integer>> it1 = Mat1.get(i).entrySet().iterator();
//                while (it1.hasNext()) {
//                    Map.Entry<String, Integer> k = it1.next();
//                 if (k.getKey().compareTo(entry.getKey())==0) {
//                        flag=1;
//                        vec.add(k.getValue());
//                        finalans.put(entry.getKey(), vec);
//                        break;
//                    } 
//                }
//                if(flag==0)
//                {
//                    vec.add(3);
//                    finalans.put(entry.getKey(), vec);
//                }
//            }
